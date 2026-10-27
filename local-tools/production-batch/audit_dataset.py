#!/usr/bin/env python3
"""Freeze dataset coverage using native receipts and fresh product content hashes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as batch


class AuditFailure(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def _require(condition, category, message):
    if not condition:
        raise AuditFailure(category, message)


def _product_paths(job, settings):
    weights = Path(settings['weights']) / job['stem']
    return {'config': weights / 'retargeter.json',
            'optimized': weights / 'optimized.done.json',
            'csv': Path(settings['output']) / f"{job['stem']}.csv",
            'done': Path(settings['artifacts']) / 'done' / f"{job['stem']}.json"}


def _file_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _checked_receipt(receipt_path, expected_key, product):
    _require(receipt_path.is_file() and product.is_file(), 'missing_product',
             f'Missing receipt/product: {receipt_path}, {product}')
    _require(not receipt_path.is_symlink() and not product.is_symlink(), 'data_alias',
             f'Symlink receipt/product: {receipt_path}, {product}')
    receipt = batch.receipt_valid(receipt_path, expected_key, product)
    _require(receipt is not None, 'receipt_or_hash_mismatch',
             f'Receipt content key or product SHA256 mismatch: {receipt_path}')
    return receipt


def _check_metadata(done, job, metadata):
    _require(done['frames'] == metadata['frames'], 'frame_metadata', 'Receipt/source frame counts differ')
    for key in ('fps', 'source_fps', 'source_frame_time'):
        _require(math.isfinite(done[key]) and math.isclose(done[key], metadata[key], rel_tol=1e-8),
                 'fps_metadata', f'Receipt/source {key} differ')
    _require(done['source_sha256'] == job['sha256'] and done['archive_member'] == job['archive_member'],
             'source_binding', 'Completion receipt does not match extraction manifest')


def _check_headers(csv_path, done):
    headers = done['headers']
    _require(isinstance(headers, list) and len(headers) == 38 and len(set(headers)) == 38
             and headers[:len(batch.ROOT_HEADER)] == batch.ROOT_HEADER
             and all(isinstance(name, str) and name for name in headers),
             'csv_headers', 'Expected native root fields plus 31 unique actuated joint fields')
    with csv_path.open(newline='') as stream:
        actual = next(csv.reader(stream), None)
    _require(actual == headers, 'csv_headers', 'CSV header does not match validated export receipt')
    return headers


def _check_source(job):
    source = Path(job['source'])
    _require(source.is_file(), 'missing_source', f'Missing source: {source}')
    _require(not source.is_symlink(), 'data_alias', f'Symlink source: {source}')
    _require(source.stat().st_size == job['bytes'], 'source_size', f'Source size changed: {source}')
    return batch.bvh_metadata(source)


def _audit_product(job, settings, metadata):
    paths = _product_paths(job, settings)
    _require(not paths['config'].parent.is_symlink(), 'data_alias', 'Per-sequence weights directory is a symlink')
    signatures = {key: _file_signature(path) for key, path in paths.items() if path.is_file()}
    optimized = _checked_receipt(paths['optimized'], batch.optimization_key(job, settings), paths['config'])
    _require(optimized['source_sha256'] == job['sha256']
             and Path(optimized['source']).resolve() == Path(job['source']).resolve(),
             'source_binding', 'Optimization receipt does not match extraction manifest')
    config = batch.load_json(paths['config'])
    _require(isinstance(config, dict), 'invalid_weights', 'Optimized configuration must be a JSON object')
    batch.verify_weights_only(config, settings['base_retargeter'])
    done = _checked_receipt(paths['done'], batch.conversion_key_for(optimized, settings), paths['csv'])
    _require(done['optimization_key'] == optimized['content_key'] and done['weights_sha256'] == optimized['sha256'],
             'weights_binding', 'Completion receipt points to different optimized weights')
    _check_metadata(done, job, metadata)
    temporal = settings['conversion_overrides'].get('temporal_ik', {})
    _require(not temporal.get('hard_velocity', False), 'hard_velocity_attestation_unavailable',
             'Native hard-velocity receipt contract is not integrated; refusing COMPLETE')
    headers = _check_headers(paths['csv'], done)
    for key, signature in signatures.items():
        _require(signature == _file_signature(paths[key]), 'concurrent_change', f'Product changed during audit: {paths[key]}')
    return {'stem': job['stem'], 'source_sha256': job['sha256'], 'csv': str(paths['csv']),
            'csv_sha256': done['sha256'], 'weights_sha256': optimized['sha256'],
            'optimization_key': optimized['content_key'], 'conversion_key': done['content_key'],
            'frames': done['frames'], 'fps': done['fps'], 'headers': headers}


def _manifest_rows(manifest, source_root):
    rows = []
    try:
        for job in batch.source_rows(manifest, source_root):
            rows.append(job)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return rows, [{'category': 'input_manifest', 'message': str(exc)}]
    return rows, []


def _unexpected_products(stems, settings):
    selections = ((Path(settings['output']), '*.csv'),
                  (Path(settings['artifacts']) / 'done', '*.json'))
    extras = []
    for directory, pattern in selections:
        extras.extend(str(path) for path in directory.glob(pattern) if path.stem not in stems)
    weights = Path(settings['weights'])
    if weights.is_dir():
        extras.extend(str(path) for path in weights.iterdir()
                      if path.is_dir() and not path.name.startswith('.') and path.name not in stems)
    return sorted(extras)


def audit_dataset(manifest, source_root, settings, expected_count=142220):
    """Return COMPLETE only for full, hash-bound coverage; never mutate products."""
    started = time.monotonic()
    manifest_signature = _file_signature(Path(manifest)) if Path(manifest).is_file() else None
    rows, failures = _manifest_rows(manifest, source_root)
    roots = [Path(source_root), Path(settings['output']), Path(settings['weights']),
             Path(settings['artifacts']), Path(settings['artifacts']) / 'done']
    aliases = [str(root) for root in roots if root.is_symlink()]
    if aliases:
        failures.append({'category': 'data_alias', 'paths': aliases})
    if len(rows) != expected_count:
        failures.append({'category': 'input_count', 'message': f'Expected {expected_count} inputs, found {len(rows)}'})
    products, expected_frames, header_reference = [], 0, None
    for job in rows:
        try:
            metadata = _check_source(job)
            expected_frames += metadata['frames']
            product = _audit_product(job, settings, metadata)
            if header_reference is None:
                header_reference = product['headers']
            _require(product.pop('headers') == header_reference, 'csv_headers', 'Dataset contains inconsistent joint headers')
            products.append(product)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'stem': job['stem'], 'category': getattr(exc, 'category', 'invalid_metadata_or_weights'),
                             'message': str(exc)})
    extras = _unexpected_products({job['stem'] for job in rows}, settings)
    if extras:
        failures.append({'category': 'unexpected_products', 'paths': extras})
    current_signature = _file_signature(Path(manifest)) if Path(manifest).is_file() else None
    if current_signature != manifest_signature:
        failures.append({'category': 'concurrent_change', 'message': 'Input manifest changed during audit'})
    complete = not failures and len(products) == expected_count
    return {'schema': 1, 'status': 'COMPLETE' if complete else 'INCOMPLETE',
            'input_manifest': str(Path(manifest).resolve()),
            'input_manifest_sha256': batch.digest(manifest) if Path(manifest).is_file() else None,
            'expected_count': expected_count, 'input_count': len(rows), 'valid_count': len(products),
            'failed_count': len(rows) - len(products), 'expected_frames': expected_frames,
            'validated_frames': sum(product['frames'] for product in products), 'headers': header_reference,
            'failure_categories': dict(Counter(failure['category'] for failure in failures)),
            'failures': failures, 'products': products, 'elapsed_seconds': time.monotonic() - started,
            'validation': {'csv_sha256_recomputed': True, 'weights_sha256_recomputed': True,
                           'finite_shape': 'Native export validation bound to receipt CSV SHA256; rows not reparsed',
                           'source_sha256': 'Bound to extraction manifest and native content keys; source size/header checked'},
            'selected_settings_key': batch.key_for(settings)}


def _check_report_destination(args, settings):
    destination = args.audit_output.resolve()
    protected_roots = [args.source_root, Path(settings['output']), Path(settings['weights']),
                       Path(settings['artifacts']) / 'done']
    _require(destination not in (args.manifest.resolve(), args.settings.resolve())
             and not any(destination.is_relative_to(root.resolve()) for root in protected_roots),
             'unsafe_report_destination', 'Audit report must not overwrite input manifests, settings, or production data')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True, help='Captured selected production settings JSON')
    parser.add_argument('--expected-count', type=int, default=142220)
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.expected_count < 1:
        parser.error('--expected-count must be positive')
    settings = batch.load_json(args.settings)
    _check_report_destination(args, settings)
    report = audit_dataset(args.manifest, args.source_root, settings, args.expected_count)
    report['selected_settings_file'] = str(args.settings.resolve())
    report['selected_settings_sha256'] = batch.digest(args.settings)
    batch.atomic_json(args.audit_output, report)
    print(f"{report['status']}: {report['valid_count']}/{report['expected_count']} clips, "
          f"{report['validated_frames']} frames; {args.audit_output}")
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
