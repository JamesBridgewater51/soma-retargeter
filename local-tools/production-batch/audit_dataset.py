#!/usr/bin/env python3
"""Freeze dataset coverage using native receipts and fresh product content hashes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import math
import os
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


def _check_export_validation(done, settings, metadata, headers):
    temporal = settings['conversion_overrides'].get('temporal_ik', settings['base_retargeter'].get('temporal_ik', {}))
    hard_selected = bool(temporal.get('hard_velocity', False))
    validation = done.get('validation')
    if validation is None and not hard_selected:
        return None  # Legacy native nonhard receipts predate the explicit validation report.
    _require(isinstance(validation, dict) and validation.get('schema') == 2
             and validation.get('frames') == metadata['frames'] and validation.get('columns') == 38,
             'export_validation', 'Missing or inconsistent native CSV validation report')
    velocity = validation.get('velocity')
    _require(isinstance(velocity, dict), 'export_validation', 'Missing native velocity validation report')
    if hard_selected:
        plan = batch.velocity_plan(settings, metadata['fps'])
        _require(plan is not None and 'velocity_scale' in plan
                 and plan['velocity_scale'] == float(temporal.get('velocity_scale', 1.0)),
                 'velocity_plan', 'Hard-selected plan must bind the explicitly selected velocity scale')
        from csv_velocity import validate_attestation
        try:
            validate_attestation(velocity, headers[len(batch.ROOT_HEADER):], metadata['frames'], plan)
        except ValueError as exc:
            raise AuditFailure('velocity_attestation', str(exc)) from exc
    else:
        _require(velocity.get('enabled') is False and velocity.get('violation_count') == 0
                 and velocity.get('checked_pairs') == 0 and velocity.get('max_ratio') is None,
                 'export_validation', 'Nonhard receipt contains inconsistent velocity validation')
    return validation


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
    headers = _check_headers(paths['csv'], done)
    validation = _check_export_validation(done, settings, metadata, headers)
    for key, signature in signatures.items():
        _require(signature == _file_signature(paths[key]), 'concurrent_change', f'Product changed during audit: {paths[key]}')
    return {'stem': job['stem'], 'source_sha256': job['sha256'], 'csv': str(paths['csv']),
            'csv_sha256': done['sha256'], 'weights_sha256': optimized['sha256'],
            'optimization_key': optimized['content_key'], 'conversion_key': done['content_key'],
            'frames': done['frames'], 'fps': done['fps'], 'headers': headers, 'validation': validation}


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
        with os.scandir(weights) as entries:
            extras.extend(entry.path for entry in entries
                          if entry.name not in stems and not entry.name.startswith('.') and entry.is_dir())
    return sorted(extras)


def _audit_job(job, settings):
    source_frames = 0
    try:
        metadata = _check_source(job)
        source_frames = metadata['frames']
        return source_frames, _audit_product(job, settings, metadata), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failure = {'stem': job['stem'], 'category': getattr(exc, 'category', 'invalid_metadata_or_weights'),
                   'message': str(exc)}
        return source_frames, None, failure


def _ordered_results(rows, settings, workers):
    inspect = partial(_audit_job, settings=settings)
    if workers == 1:
        yield from map(inspect, rows)
        return
    # Bound queued futures while retaining input-order aggregation on Python
    # versions whose Executor.map does not yet support a buffersize argument.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='dataset-audit') as executor:
        for start in range(0, len(rows), workers * 8):
            yield from executor.map(inspect, rows[start:start + workers * 8])


def audit_dataset(manifest, source_root, settings, expected_count=142220, *, workers=1):
    """Return COMPLETE only for full, hash-bound coverage; never mutate products."""
    if workers < 1:
        raise ValueError('Audit workers must be positive')
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
    for source_frames, product, failure in _ordered_results(rows, settings, workers):
        expected_frames += source_frames
        if failure is not None:
            failures.append(failure)
            continue
        if header_reference is None:
            header_reference = product['headers']
        if product.pop('headers') != header_reference:
            failures.append({'stem': product['stem'], 'category': 'csv_headers',
                             'message': 'Dataset contains inconsistent joint headers'})
            continue
        products.append(product)
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
            'audit_workers': workers,
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
    parser.add_argument('--workers', type=int, default=1, help='Read-only per-clip audit threads; results retain manifest order')
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.expected_count < 1 or args.workers < 1:
        parser.error('--expected-count and --workers must be positive')
    settings = batch.load_json(args.settings)
    _check_report_destination(args, settings)
    report = audit_dataset(args.manifest, args.source_root, settings, args.expected_count, workers=args.workers)
    report['selected_settings_file'] = str(args.settings.resolve())
    report['selected_settings_sha256'] = batch.digest(args.settings)
    batch.atomic_json(args.audit_output, report)
    print(f"{report['status']}: {report['valid_count']}/{report['expected_count']} clips, "
          f"{report['validated_frames']} frames; {args.audit_output}")
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
