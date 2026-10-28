"""Read-only verification of the translation correction's effect on cached ranking inputs."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path('/data/liuminghao')
ART = ROOT / 'artifacts/soma-retargeter/agibot-x2ultra/challenge100-review/selection'
FEATURES = ['pelvis_speed_p95_mps', 'pelvis_vertical_span_m', 'torso_tilt_p95_deg',
            'joint_angular_speed_p95_radps', 'hand_near_footbaseline_fraction']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_functions():
    tree = ast.parse((ART / 'select_challenge100.py').read_text())
    namespace = {'np': np, 'pd': pd, 'Rotation': Rotation, 're': re}
    definitions = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'main']
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    quota = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'QUOTAS' for t in n.targets))
    namespace['QUOTAS'] = ast.literal_eval(quota.value)
    metric = next(n for n in definitions if n.name == 'bvh_metrics')
    for node in ast.walk(metric):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 3:
            node.value.elts.extend([ast.Name(id='joints', ctx=ast.Load()), ast.parse('orientations[0].as_matrix()', mode='eval').body])
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), '<actual-selector-functions>', 'exec'), namespace)
    return namespace, main


def extracted_function(namespace, name, arguments, statements, returned):
    function = ast.parse('def placeholder(' + ','.join(arguments) + '):\n pass').body[0]
    function.name = name
    function.body = copy.deepcopy(statements) + [ast.Return(value=ast.Name(id=returned, ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, '<actual-selector-block>', 'exec'), namespace)
    return namespace[name]


def inspect_candidate(row):
    name = row['filename']
    metrics, names, positions, joints, root_rotations = MEASURE(ROOT / 'datasets/bones-seed/raw/soma-uniform/full' / (name + '.bvh'), True)
    translated = [j['name'] for j in joints if any(c.endswith('position') for c in j['channels'])]
    assert translated == ['Root', 'Hips'] and joints[names.index('Hips')]['parent'] == 0
    shift = joints[0]['offset'][None, :] / 100 + np.einsum('tij,j->ti', root_rotations, joints[names.index('Hips')]['offset'] / 100)
    variation = float(np.max(np.abs(shift - shift[:1])))
    old = dict(row['metrics'])
    if 'pelvis_tilt_p95_deg' in old:
        old['torso_tilt_p95_deg'] = old.pop('pelvis_tilt_p95_deg')
    differences = {key: abs(old[key] - metrics[key]) for key in FEATURES + ['pelvis_height_p05_m']}
    return {'filename': name, 'metrics': metrics, 'old_metrics': old, 'rank_feature_abs_differences': differences,
            'extra_world_translation_variation_m': variation, 'predicted_baseline_shift_m': float(shift[0, 1]),
            'actual_baseline_shift_m': old['observed_footbaseline_m'] - metrics['observed_footbaseline_m']}


def replay(selected_rows, original, selector, rounding=None):
    copied = copy.deepcopy(selected_rows)
    if rounding is not None:
        for row in copied:
            for key in FEATURES:
                row['metrics'][key] = round(row['metrics'][key], rounding)
    return [row['filename'] for row in selector(copied, original)]


started = time.monotonic()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
output = ART / 'ranking-verification.json'
if output.exists():
    raise FileExistsError(output)
proposal = json.loads((ART / 'challenge100-proposal.json').read_text())
frozen_sha = sha(ART / 'challenge100-proposal.json')
assert frozen_sha == '0cd268a5436d7cedf538f7cf417d77571ec7fef5c7cbfe0776ca397bbadea778'
coverage = json.loads((ART / 'pool-coverage.json').read_text())
metadata_path = ROOT / 'datasets/bones-seed/metadata/seed_metadata_v004.parquet'
assert sha(metadata_path) == coverage['metadata_sha256']
frame = pd.read_parquet(metadata_path).fillna('')
assert coverage['threeway_available'] == len(frame) == 142220
for field in ('soma_available', 'x2_available', 'g1_available'):
    frame[field] = True
namespace, main_node = source_functions()
MEASURE = namespace['bvh_metrics']
make_candidates = extracted_function(namespace, 'make_candidates', ['df', 'keep'],
    [n for n in main_node.body if 159 <= n.lineno <= 161 or 175 <= n.lineno <= 192], 'candidates')
selector = extracted_function(namespace, 'replay_selector', ['measured', 'keep'],
    [n for n in main_node.body if 216 <= n.lineno <= 257], 'selected')
original = {r['filename'] for r in proposal if r['original11']}
candidates = make_candidates(frame, original)
assert len(candidates) == len({r['filename'] for r in candidates}) == 481
cache_path = ART / 'candidate-metrics-anatomical-axis.jsonl'
cache = {r['filename']: r for r in map(json.loads, cache_path.read_text().splitlines()) if not r.get('error')}
assert {r['filename'] for r in candidates} <= set(cache)
print(json.dumps({'event': 'RANKING_VERIFY_START', 'ranking_candidates': len(candidates), 'cached_rows_checked': len(cache), 'workers': 8}), flush=True)
corrected = []
with ThreadPoolExecutor(max_workers=8) as executor:
    for index, checked in enumerate(executor.map(inspect_candidate, cache.values()), 1):
        corrected.append(checked)
        if index % 50 == 0:
            print(json.dumps({'event': 'RANKING_VERIFY_PROGRESS', 'checked': index, 'total': len(cache)}), flush=True)
by_name = {r['filename']: r for r in corrected}
old_rows, new_rows = [], []
for row in candidates:
    checked = by_name[row['filename']]
    old_rows.append({**row, 'metrics': checked['old_metrics'], 'error': None})
    new_rows.append({**row, 'metrics': checked['metrics'], 'error': None})
old_selection = replay(old_rows, original, selector)
new_selection = replay(new_rows, original, selector)
old_rounded = replay(old_rows, original, selector, 8)
new_rounded = replay(new_rows, original, selector, 8)
frozen_names = {r['filename'] for r in proposal}
max_differences = {key: max(r['rank_feature_abs_differences'][key] for r in corrected) for key in FEATURES + ['pelvis_height_p05_m']}
max_variation = max(r['extra_world_translation_variation_m'] for r in corrected)
noninvariant = [r['filename'] for r in corrected if r['extra_world_translation_variation_m'] > 1e-10 or any(v > 1e-8 for v in r['rank_feature_abs_differences'].values())]
report = {'status': 'VERIFIED' if not noninvariant else 'DIFFERENCES_FOUND', 'selection_sha256': frozen_sha,
    'selection_unchanged': sha(ART / 'challenge100-proposal.json') == frozen_sha,
    'ranking_candidates': len(candidates), 'cached_candidates_recomputed': len(corrected),
    'selector_sha256': sha(ART / 'select_challenge100.py'), 'old_cache_sha256': sha(cache_path),
    'translation_channel_joints': ['Root', 'Hips'], 'proof': 'The erroneous additions produce Root.OFFSET + Root.world_R(t) @ Hips.OFFSET for every global joint; this was measured constant over all sampled frames of every cached candidate. Relative heights, differences, ranges and orientation/rotation features cancel this constant. Only absolute observed foot baseline is shifted.',
    'maximum_extra_world_translation_variation_m': max_variation,
    'maximum_rank_feature_absolute_differences': max_differences, 'noninvariant_candidates': noninvariant,
    'unrounded_replay_old_matches_frozen_set': set(old_selection) == frozen_names,
    'unrounded_replay_new_matches_old_set': set(new_selection) == set(old_selection),
    'unrounded_replay_new_matches_frozen_set': set(new_selection) == frozen_names,
    'unrounded_old_new_added': sorted(set(new_selection) - set(old_selection)),
    'unrounded_old_new_removed': sorted(set(old_selection) - set(new_selection)),
    'rounded_1e8_replay_old_matches_new_set': set(old_rounded) == set(new_rounded),
    'rounded_1e8_replay_old_matches_frozen_set': set(old_rounded) == frozen_names,
    'scope_limit': 'This verifies the translation correction effect on the actual shortlist ranking, not a globally hardest100 theorem. Semantic quotas, family deduplication and mandatory originals remain selection constraints. Proxies are not contact truth.',
    'elapsed_seconds': time.monotonic() - started, 'clips': corrected}
assert report['selection_unchanged']
with output.open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({k:v for k,v in report.items() if k != 'clips'}), flush=True)
