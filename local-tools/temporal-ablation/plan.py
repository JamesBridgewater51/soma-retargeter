"""Prepare a small controlled screen or full confirmation matrix; never launch jobs."""
import argparse
import json
import os
from pathlib import Path
import sys


def configurations(scales):
    variants = {'baseline': {'temporal_ik': {'enabled': False}}}
    for scale in scales:
        for family in ('smooth', 'limits', 'combined'):
            variants[f'{family}-{scale:g}'] = {'temporal_ik': {
                'enabled': True,
                'mode': 'prediction',
                'weight': scale if family in ('smooth', 'combined') else 0.0,
                'hard_velocity': family in ('limits', 'combined'),
                'velocity_scale': 1.0,
                'limit_margin': 0.05,
                'limit_weight': 0.0,
                'velocity_avoidance_weight': scale if family in ('limits', 'combined') else 0.0,
                'velocity_margin_fraction': 0.2,
            }}
    return variants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage-root', type=Path, default=Path(os.environ.get('STORAGE_ROOT', '/data/liuminghao')))
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--motions', nargs='+', required=True, help='Exact motion names or all')
    parser.add_argument('--weight-sets', nargs='+', choices=('default', 'optimized'), default=['optimized'])
    parser.add_argument('--scales', nargs='+', type=float, default=[0.25, 1.0, 4.0])
    parser.add_argument('--variants', nargs='+', help='Restrict full confirmation to screened variant names plus baseline')
    parser.add_argument('--combined-weights', nargs=2, type=float, metavar=('SMOOTH', 'VELOCITY_AVOIDANCE'),
        help='Add independently screened raw-angular smoothness and dimensionless velocity-avoidance weights')
    return parser.parse_args()


def main():
    args = parse_args()
    if any(not 0 < scale < float('inf') for scale in args.scales) or len(set(args.scales)) != len(args.scales):
        raise ValueError('Screen scales must be distinct, finite, and positive')
    selection = json.loads((args.storage_root / 'datasets/bones-seed/metadata/x2-validation-selection.json').read_text())
    known = [entry['name'] for entry in selection]
    motions = known if args.motions == ['all'] else args.motions
    if len(set(motions)) != len(motions) or set(motions) - set(known):
        raise ValueError('Unknown or duplicate selected motion')
    variants = configurations(args.scales)
    if args.combined_weights:
        smooth, near_limit = args.combined_weights
        if any(not 0 < weight < float('inf') for weight in (smooth, near_limit)):
            raise ValueError('Combined weights must be positive and finite')
        variants[f'combined-s{smooth:g}-l{near_limit:g}'] = {'temporal_ik': {
            **variants[f'combined-{args.scales[0]:g}']['temporal_ik'],
            'weight': smooth, 'velocity_avoidance_weight': near_limit,
        }}
    if args.variants:
        selected = set(args.variants) | {'baseline'}
        if selected - set(variants):
            raise ValueError('Unknown variant; preserve the screen scales during confirmation')
        variants = {name: config for name, config in variants.items() if name in selected}
    args.plan_dir.mkdir(parents=True, exist_ok=False)
    (args.plan_dir / 'selection.json').write_text(json.dumps([entry for entry in selection if entry['name'] in motions], indent=2))
    commands = []
    for name, config in variants.items():
        config_path = args.plan_dir / (name + '.json')
        config_path.write_text(json.dumps(config, indent=2, allow_nan=False))
        for weight_set in args.weight_sets:
            for motion in motions:
                commands.append([args.python, str(Path(__file__).with_name('run.py').resolve()),
                    '--motion', motion, '--variant', name, '--weight-set', weight_set,
                    '--temporal-config', str(config_path.resolve()),
                    '--storage-root', str(args.storage_root),
                    '--output-dir', str(args.results_dir / weight_set / name / motion)])
    manifest = {
        'commands': commands, 'launch_authorized': False,
        'outer_pipeline': 'native post/contact processing unchanged; no collision objectives added',
        'grid_units': 'smoothness multiplies raw angular prediction error; velocity avoidance multiplies dimensionless excess of deltaq/(URDFvmax*dt) beyond 0.8; each weight is squared in least squares',
        'extra_position_limit_weight': 0.0, 'velocity_margin_fraction': 0.2, 'velocity_scale': 1.0,
        'selection_role': 'exploratory screening; choose family finalists then confirm on all eleven fixed-weight clips',
        'weight_set_policy': 'optimized fixed per-sequence and native default are separate strata; never pool',
    }
    (args.plan_dir / 'commands.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f'PLAN_READY {len(commands)} explicit commands; none executed')


if __name__ == '__main__':
    main()
