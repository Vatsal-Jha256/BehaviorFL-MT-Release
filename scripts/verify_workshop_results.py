#!/usr/bin/env python3
"""Check saved runs against release_manifest.json."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_run(data: dict, run: dict, manifest: dict) -> None:
    def require(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(message)

    require(data['method'] == run['method'] and data['seed'] == run['seed'], 'method/seed mismatch')
    require(data['n_rounds'] == manifest['rounds'], 'round count mismatch')
    require(data['participation_fraction'] == manifest['participation_fraction'], 'participation mismatch')
    require(data['dataset_metadata']['schema'] == manifest['dataset_schema'], 'dataset schema mismatch')
    require(data['dataset_n_rows'] == manifest['dataset_rows'], 'dataset row count mismatch')
    require(data['dataset_n_fl_clients'] == manifest['training_clients'], 'training-client count mismatch')
    for key, value in run['method_config'].items():
        require(data['method_config'].get(key) == value, f'method config mismatch: {key}')
    held = data['heldout_evaluation']
    require(held['n_held_out_vehicles'] == manifest['heldout_clients'], 'held-out client count mismatch')
    for section, metric, count in [('destination', 'macro_f1', 'destination_eval_trips'),
                                    ('destination', 'hr_at_5', 'destination_eval_trips'),
                                    ('state_natural_sensors', 'auprc', 'state_eval_trips'),
                                    ('state_gps_only', 'auprc', 'state_eval_trips')]:
        row = held[section]
        require(row['n'] == manifest[count], f'{section}: evaluation trip count mismatch')
        value = row[metric]
        require(math.isfinite(value) and 0 <= value <= 1, f'{section}.{metric}: invalid metric')
    for key in ['destination_macro_f1', 'destination_hr5', 'state_auprc_natural', 'state_auprc_gps']:
        series = data['periodic_heldout_trajectory'][key]
        require([row[0] for row in series] == manifest['periodic_rounds'], f'{key}: missing/misaligned trajectory')
        require(all(math.isfinite(row[1]) and 0 <= row[1] <= 1 for row in series), f'{key}: invalid trajectory')
        section, metric = {'destination_macro_f1': ('destination', 'macro_f1'),
                           'destination_hr5': ('destination', 'hr_at_5'),
                           'state_auprc_natural': ('state_natural_sensors', 'auprc'),
                           'state_auprc_gps': ('state_gps_only', 'auprc')}[key]
        # Batch mean-fill depends on the inference batch composition. Periodic
        # and final evaluation use different batching; only batch-independent
        # fills should produce identical final-round scores.
        if run['method_config'].get('fill_mode') != 'batch_mean':
            require(abs(series[-1][1] - held[section][metric]) < 1e-6, f'{key}: final trajectory differs from checkpoint evaluation')
    if run['method_config']['strategy'] == 'task_aware_fedavg':
        discovered = data['known_state_clients']
        require(len(discovered) == manifest['rounds'], 'missing discovery history')
        require(all(a <= b for a, b in zip(discovered, discovered[1:])), 'discovery count decreased')
        require(all(0 <= n <= 188 for n in discovered), 'invalid discovery count')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'release_manifest.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    errors = []
    for run in manifest['runs']:
        stem = f"{run['method']}_seed{run['seed']}"
        try:
            verify_run(json.loads((args.results_dir / f'{stem}.json').read_text()), run, manifest)
            if not (args.results_dir / f'{stem}.pt').is_file():
                raise ValueError('missing checkpoint')
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f'{stem}: {exc}')
    for error in errors:
        print(f'[verify] FAIL {error}')
    print(f"[verify] {'FAIL' if errors else 'PASS'}: {len(manifest['runs'])} saved runs; {len(errors)} errors")
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
