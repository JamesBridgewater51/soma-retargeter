"""Auditable diversity-first challenge proposal; reads upstream data, writes owned artifacts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

QUOTAS = {'acrobatics': 7, 'hand_support': 8, 'floor_recovery': 16,
          'seated': 8, 'terrain': 18, 'object_interaction': 16,
          'agility_sport': 10, 'dance': 8, 'rare_actions': 9}


def family(row):
    text = str(row['content_name']).lower()
    text = re.sub(r'_(?:start|stop|loop|idle|very_fast|fast|slow|[rl]|\d+)(?=_|$)', '', text)
    return re.sub('_+', '_', text).strip('_')


def classify(row):
    name = str(row['content_name']).lower()
    body = str(row['content_body_position']).lower()
    prop = str(row['content_props']).lower()
    move = str(row['content_type_of_movement']).lower()
    cat = str(row['category']).lower()
    text = ' '.join(str(row[c]).lower() for c in ['content_name', 'content_short_description', 'content_technical_description'])
    if re.search(r'\b(stairs|ladder|obstacle|box)\b', prop):
        return 'terrain'
    if any(x in text for x in ['handstand', 'cartwheel', 'push-up', 'push up', 'inchworm', 'plank', 'spider crawl']) or 'handstand' in body:
        return 'hand_support'
    if any(x in text for x in ['flip', 'somersault', 'fire_in_the_hole', 'back handspring']):
        return 'acrobatics'
    if (any(x in text for x in ['lying', 'faint', 'falling', 'stomach', 'crawl', 'all fours', 'kneeling', 'get up'])
            or 'rolling' in move or re.search(r'(^|_)(roll|lying|faint|crawl|kneeling|stomach)(_|$)', name)
            or any(x in body for x in ['lying', 'crawling', 'all fours', 'kneeling'])):
        return 'floor_recovery'
    if 'sitting' in body or any(x in name for x in ['crossed_legs', 'sit_up', 'sitting']):
        return 'seated'
    if prop not in ['', '0', 'none', 'nan']:
        return 'object_interaction'
    if cat == 'dancing' and ('fast' in name or any(x in text for x in ['spin', 'jump', 'turn', 'kick', 'break'])):
        return 'dance'
    if cat in ['martial arts', 'magic', 'stunts', 'other', 'complex actions']:
        return 'rare_actions'
    if cat == 'sports' or any(x in text for x in ['sprint', 'broad jump', 'leap', 'dodge', 'jumping', 'running']) or 'fast' in name:
        return 'agility_sport'
    return None


def prior(row):
    text = ' '.join(str(row[c]).lower() for c in ['content_name', 'content_short_description', 'content_body_position', 'content_technical_description'])
    return (sum(word in text for word in ['fast', 'jump', 'flip', 'handstand', 'roll', 'lying', 'stomach', 'back', 'slip', 'slide', 'throw', 'heavy', 'kick', 'spin', 'crawl', 'floor', 'run', 'climb'])
            + float(row['content_complex_action']) + float(row['content_vertical_move']))


def bvh_metrics(path, return_positions=False):
    # Parse hierarchy as data; no downloaded Python modules are imported.
    joints, stack, current, end_site = [], [], None, False
    with path.open() as handle:
        for line in handle:
            words = line.strip().split()
            if not words:
                continue
            if words[0] == 'MOTION':
                break
            if words[0] in ['ROOT', 'JOINT']:
                current = len(joints)
                joints.append({'name': words[1], 'parent': stack[-1] if stack else -1})
                end_site = False
            elif words[0] == 'End':
                end_site = True
            elif words[0] == '{':
                stack.append(None if end_site else current)
            elif words[0] == '}':
                stack.pop()
                current = stack[-1] if stack else None
                end_site = current is None and bool(stack)
            elif words[0] == 'OFFSET' and current is not None and not end_site:
                joints[current]['offset'] = np.array([float(v) for v in words[1:4]])
            elif words[0] == 'CHANNELS' and current is not None:
                joints[current]['channels'] = words[2:]
        frames = int(next(handle).split(':')[1])
        frame_time = float(next(handle).split(':')[1])
        rows = [np.fromstring(line, sep=' ') for i, line in enumerate(handle) if i % 4 == 0 and line.strip()]
    values = np.stack(rows)
    dt = frame_time * 4
    positions, orientations, cursor, rates = [], [], 0, []
    for joint in joints:
        channels = joint['channels']
        data = values[:, cursor:cursor+len(channels)]
        cursor += len(channels)
        local_position = np.broadcast_to(joint['offset'], (len(values), 3)).copy()
        axes, angles = [], []
        for i, channel in enumerate(channels):
            if channel.endswith('position'):
                local_position[:, 'XYZ'.index(channel[0])] = data[:, i]
            elif channel.endswith('rotation'):
                axes.append(channel[0]); angles.append(data[:, i])
        local_rotation = Rotation.from_euler(''.join(axes), np.stack(angles, axis=1), degrees=True)
        angle_rate = (local_rotation[:-1].inv() * local_rotation[1:]).magnitude() / dt
        rates.append(angle_rate)
        parent = joint['parent']
        if parent < 0:
            positions.append(local_position / 100)
            orientations.append(local_rotation)
        else:
            positions.append(positions[parent] + orientations[parent].apply(local_position / 100))
            orientations.append(orientations[parent] * local_rotation)
    names = [j['name'] for j in joints]
    pelvis_index = names.index('Hips')
    pelvis = positions[pelvis_index]
    torso = positions[names.index('Chest')] - pelvis
    torso /= np.linalg.norm(torso, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(torso[:, 1], -1, 1)))
    # Relative to observed minimum foot elevation, not an asserted scene floor.
    foot_indices = [i for i, name in enumerate(names) if any(t in name.lower() for t in ['foot', 'toe'])]
    hand_indices = [i for i, name in enumerate(names) if 'hand' in name.lower() and not any(t in name.lower() for t in ['thumb', 'index', 'middle', 'ring', 'pinky'])]
    feet_y = np.stack([positions[i][:, 1] for i in foot_indices])
    baseline = float(np.percentile(feet_y, 1))
    hands_y = np.min(np.stack([positions[i][:, 1] for i in hand_indices]), axis=0)
    result = {'bvh_frames': frames, 'bvh_frame_time': frame_time, 'sample_hz': 1 / dt,
            'pelvis_speed_p95_mps': float(np.percentile(np.linalg.norm(np.diff(pelvis, axis=0), axis=1) / dt, 95)),
            'pelvis_height_p05_m': float(np.percentile(pelvis[:, 1] - baseline, 5)),
            'pelvis_vertical_span_m': float(np.ptp(pelvis[:, 1])),
            'torso_tilt_p95_deg': float(np.percentile(tilt, 95)),
            'hand_near_footbaseline_fraction': float(np.mean(hands_y - baseline < .15)),
            'joint_angular_speed_p95_radps': float(np.percentile(np.stack(rates), 95)),
            'observed_footbaseline_m': baseline}
    if return_positions:
        return result, names, np.stack(positions, axis=1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage-root', type=Path, default=Path('/data/liuminghao'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'challenge100-proposal.json').exists():
        raise RuntimeError('Selection is frozen; use a new output directory for any exploratory reranking.')
    root = args.storage_root
    metadata_path = root/'datasets/bones-seed/metadata/seed_metadata_v004.parquet'
    df = pd.read_parquet(metadata_path).fillna('')
    original = json.loads(args.original.read_text())
    keep = {r['name'] for r in original}
    source = root/'datasets/bones-seed/raw/soma-uniform/full'
    x2 = root/'datasets/bones-seed/converted/agibot-x2ultra/per-sequence-optimized'
    g1 = root/'datasets/bones-seed/raw/g1/csv'
    source_names = set(os.listdir(source)); x2_names = set(os.listdir(x2))
    g1_members = {f'{d}/{n}' for d in os.listdir(g1) for n in os.listdir(g1/d)}
    df['bucket'] = df.apply(classify, axis=1)
    df['family'] = df.apply(family, axis=1)
    df['semantic_prior'] = df.apply(prior, axis=1)
    df['soma_available'] = df.filename.map(lambda n: n+'.bvh' in source_names)
    df['x2_available'] = df.filename.map(lambda n: n+'.csv' in x2_names)
    df['g1_available'] = df.move_g1_path.map(lambda p: p.removeprefix('g1/csv/') in g1_members)
    coverage = {'pool_count': len(df), 'nonmirror_count': int((~df.is_mirror).sum()),
                'soma_available': int(df.soma_available.sum()), 'official_g1_available': int(df.g1_available.sum()),
                'production_x2_available': int(df.x2_available.sum()),
                'threeway_available': int((df.soma_available & df.g1_available & df.x2_available).sum()),
                'bucket_pool_counts': df.bucket.value_counts().to_dict(), 'quota': QUOTAS,
                'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                'missing_requested_categories': ['cycling: supine ab_bicycle exercise exists, but no bicycle-riding evidence found', 'offroad: no offroad terrain label found'],
                'scene_geometry_status': 'No prop/terrain scene geometry is supplied by these BVH/CSV files. Semantic interactions only; meshes must not be fabricated.'}
    (args.output/'pool-coverage.json').write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage), flush=True)
    eligible = df[(~df.is_mirror) & df.soma_available & df.g1_available & df.x2_available]
    candidates = []
    for bucket in QUOTAS:
        subset = eligible[eligible.bucket == bucket].sort_values(['semantic_prior', 'filename'], ascending=[False, True])
        # Round-robin secondary labels prevents plentiful heavy-object variants or
        # complex walk/gesture compounds from hiding rare props/martial arts.
        subset = subset.drop_duplicates('family')
        column = {'rare_actions': 'category', 'object_interaction': 'content_props',
                  'floor_recovery': 'content_body_position', 'seated': 'content_body_position',
                  'agility_sport': 'category'}.get(bucket)
        if column:
            groups = [group.to_dict('records') for _, group in subset.groupby(column, sort=True)]
            diverse = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
            candidates.extend(diverse[:65])
        else:
            candidates.extend(subset.head(65).to_dict('records'))
    chosen_names = {r['filename'] for r in candidates}
    candidates.extend(df[df.filename.isin(keep - chosen_names)].to_dict('records'))
    metrics_cache = args.output/'candidate-metrics-native-translation.jsonl'
    cached = {}
    if metrics_cache.exists():
        for line in metrics_cache.read_text().splitlines():
            row = json.loads(line); cached[row['filename']] = row
    measured = []
    with metrics_cache.open('a') as handle:
        for index, row in enumerate(candidates):
            name = row['filename']
            if name in cached:
                row['metrics'] = dict(cached[name]['metrics'])
                if 'pelvis_tilt_p95_deg' in row['metrics']:
                    row['metrics']['torso_tilt_p95_deg'] = row['metrics'].pop('pelvis_tilt_p95_deg')
                row['error'] = cached[name]['error']
                measured.append(row); continue
            try:
                row['metrics'] = bvh_metrics(source/(name+'.bvh'))
                row['error'] = None
            except Exception as exc:
                row['error'] = str(exc)
            handle.write(json.dumps(row, default=str)+'\n'); handle.flush()
            measured.append(row)
            if index % 25 == 0: print('measured', index, len(candidates), name, row['error'], flush=True)
    valid = [r for r in measured if not r.get('error')]
    for bucket in QUOTAS:
        subset = [r for r in valid if r['bucket'] == bucket]
        for key in ['pelvis_speed_p95_mps', 'pelvis_vertical_span_m', 'torso_tilt_p95_deg', 'joint_angular_speed_p95_radps', 'hand_near_footbaseline_fraction']:
            ordered = sorted(subset, key=lambda r: r['metrics'][key])
            for i, row in enumerate(ordered):
                row.setdefault('severity_percentiles', {})[key] = i / max(1, len(ordered)-1)
        for row in subset:
            p = row['severity_percentiles']
            if bucket in ['floor_recovery', 'hand_support', 'seated']:
                weights = [0.1, 0.15, 0.3, 0.15, 0.3]
            else:
                weights = [0.3, 0.2, 0.2, 0.25, 0.05]
            row['severity_score'] = sum(w*p[k] for w,k in zip(weights, p))
    selected = [r for r in valid if r['filename'] in keep]
    assert {r['filename'] for r in selected} == keep, 'An original motion failed numerical parsing/availability'
    used = {r['family'] for r in selected}
    # Semantic coverage guarantees precede within-category numerical ranking.
    anchors = {'terrain': ['stairs_climbing_jog_down', 'stairs_climbing_jog_up', 'stairs_slip',
                           'stair_slide_down', 'ladder_slide_down', 'ladder_slip',
                           'ladder_climbing_up_asymmetrical', 'ladder_climbing_down_through',
                           'box_hspu', 'jump_from_box_to_safety_roll', 'vault_over'],
               'floor_recovery': ['stand_up_lying_stomach', 'faint_stand_up_lying_stomach',
                                  'ab_bicycle', 'scissor_kick', 'postmortem_convulsions',
                                  'crawl_ff', 'tiger_jump_to_shoulder_roll', 'kneeling'],
               'hand_support': ['cartwheel', 'handstand_jump', 'push_up', 'elbow_plank'],
               'rare_actions': ['ib_dodge_up', 'ib_dodge_back', 'ib_combat_turn_jog',
                                'idle_right_to_spell_two_hands', 'burning'],
               'agility_sport': ['broad_jump']}
    for bucket, families in anchors.items():
        for target in families:
            options = [r for r in valid if r['bucket'] == bucket and r['family'] == target and r['family'] not in used]
            if options:
                row = max(options, key=lambda r: r['severity_score'])
                selected.append(row); used.add(row['family'])
    for bucket, quota in QUOTAS.items():
        need = quota - sum(r['bucket'] == bucket for r in selected)
        for row in sorted([r for r in valid if r['bucket'] == bucket and r['filename'] not in keep], key=lambda r: (-r['severity_score'], r['filename'])):
            if need <= 0: break
            if row['family'] in used: continue
            selected.append(row); used.add(row['family']); need -= 1
        if need: raise RuntimeError(f'Insufficient distinct {bucket}: need {need} more')
    manifest_path = root/'artifacts/soma-retargeter/x2-production/full-input-manifest.jsonl'
    names = {r['filename'] for r in selected}
    manifest = {}
    for line in manifest_path.open():
        row = json.loads(line)
        stem = Path(row['output_filename']).stem
        if stem in names: manifest[stem] = row
    for rank, row in enumerate(selected, 1):
        name = row['filename']; identity = manifest[name]
        row.update({'review_index': rank, 'original11': name in keep, 'fps': 120,
                    'frames': identity['frames'], 'duration_s': identity['frames']/120,
                    'soma_bvh': str(source/(name+'.bvh')), 'official_g1_csv': str(root/'datasets/bones-seed/raw'/row['move_g1_path']),
                    'production_x2_csv': str(x2/(name+'.csv')), 'source_manifest': identity,
                    'geometry_provenance': 'semantic interaction only; no prop/terrain mesh in motion files'})
        # Validate CSV rows/schema and finite payload for chosen motions only.
        row['csv_validation'] = {}
        for label, path in [('official_g1',row['official_g1_csv']),('production_x2',row['production_x2_csv'])]:
            values = np.loadtxt(path, delimiter=',', skiprows=1)
            row['csv_validation'][label] = {'frames': len(values), 'columns': values.shape[1], 'finite': bool(np.isfinite(values).all()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            assert len(values) == row['frames'], (name, label, 'frame count mismatch')
            assert np.isfinite(values).all(), (name, label, 'nonfinite CSV')
            assert np.array_equal(values[:, 0], np.arange(row['frames'])), (name, label, 'frame index mismatch')
        assert row['metrics']['bvh_frames'] == row['frames'], (name, 'BVH frame count mismatch')
        actual_hash = hashlib.sha256(Path(row['soma_bvh']).read_bytes()).hexdigest()
        assert actual_hash == identity['sha256'], (name, 'source manifest hash mismatch')
        row['source_sha256_verified'] = actual_hash
    assert len(selected) == 100 and len(names) == 100
    (args.output/'challenge100-proposal.json').write_text(json.dumps(selected, indent=2, default=str))
    pd.DataFrame([{k:r[k] for k in ['review_index','filename','bucket','family','original11','frames','fps','duration_s','content_props','content_body_position','content_short_description','severity_score','soma_bvh','official_g1_csv','production_x2_csv']} for r in selected]).to_csv(args.output/'challenge100-proposal.csv', index=False)
    audit = {'selected_count': len(selected), 'original11_preserved': len(keep), 'distribution': dict(Counter(r['bucket'] for r in selected)),
             'candidate_count': len(candidates), 'measured_success': len(valid), 'measured_errors': len(measured)-len(valid),
             'unique_semantic_families': len({r['family'] for r in selected}),
             'total_frames': sum(r['frames'] for r in selected), 'total_duration_s': sum(r['duration_s'] for r in selected),
             'claim': 'Diverse high-challenge proposal, NOT a mathematically unique hardest100. Numerical percentiles are within shortlisted category, not full-pool extrema.',
             'cautions': ['30 Hz subsampling for severity ranking; full120Hz files retained', 'Foot-height baseline is observed kinematic proxy, not known terrain height', 'Non-foot support is a kinematic/label hypothesis, not force/contact proof', 'Scene objects/stairs/ladder geometry absent from motion files', 'Original11 preserved even when multiple rotations belong to one action family'],
             'csv_validation': {k: {'files': len(selected), 'columns': sorted({r['csv_validation'][k]['columns'] for r in selected}), 'all_finite': all(r['csv_validation'][k]['finite'] for r in selected), 'all_frames_match': all(r['csv_validation'][k]['frames'] == r['frames'] for r in selected)} for k in ['official_g1', 'production_x2']},
             'source_hashes_verified': len(selected),
             'metric_definitions': {'torso_tilt_p95_deg': 'Angle of geometric Hips-to-Chest vector from world +Y; not raw joint Euler axes', 'hand_near_footbaseline_fraction': 'Fraction with either hand below observed foot-height first percentile plus15cm; kinematic proxy only'}}
    (args.output/'selection-audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
