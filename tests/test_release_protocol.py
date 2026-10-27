"""Tests for saved-run validation, sampling, and resumption."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from experiments.run_behaviorfl_experiments import TaskAwareFedAvg, _load_existing_result
from scripts.verify_workshop_results import verify_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_run():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    run = next(r for r in manifest['runs'] if r['method'] == 'behaviorfl_mt' and r['seed'] == 7)
    data = json.loads((ROOT / 'results/behaviorfl_mt_seed7.json').read_text())
    return manifest, run, data


@pytest.mark.parametrize('corruption', ['seed', 'rounds', 'mask_config', 'trajectory', 'nonfinite'])
def test_verifier_rejects_invalid_evidence(frozen_run, corruption):
    manifest, run, data = frozen_run
    data = copy.deepcopy(data)
    if corruption == 'seed':
        data['seed'] = 123
    elif corruption == 'rounds':
        data['n_rounds'] = 3
    elif corruption == 'mask_config':
        data['method_config']['sensor_block_dropout_p'] = 0
    elif corruption == 'trajectory':
        data['periodic_heldout_trajectory']['state_auprc_natural'] = []
    else:
        data['heldout_evaluation']['state_gps_only']['auprc'] = float('nan')
    with pytest.raises(ValueError):
        verify_run(data, run, manifest)


def test_resume_requires_requested_periodic_evaluation(tmp_path, frozen_run):
    _, _, data = frozen_run
    path = tmp_path / 'behaviorfl_mt_seed7.json'
    path.write_text(json.dumps(data))
    args = (tmp_path, 'behaviorfl_mt', 7, 60, 0.02, data['dataset_metadata'])
    assert _load_existing_result(*args, eval_every_n_rounds=5) is not None
    data['periodic_heldout_trajectory'] = {}
    path.write_text(json.dumps(data))
    assert _load_existing_result(*args, eval_every_n_rounds=5) is None
    assert _load_existing_result(*args, eval_every_n_rounds=0) is not None


def test_discovery_sampler_respects_cohort_and_available_clients():
    clients = {str(i): SimpleNamespace(cid=str(i)) for i in range(10)}
    manager = SimpleNamespace(num_available=lambda: len(clients), all=lambda: clients)
    strategy = TaskAwareFedAvg(fraction_fit=0.4, min_fit_clients=4,
                              min_available_clients=4, state_clients_per_round=2)
    parameters = ndarrays_to_parameters([np.zeros(1, dtype=np.float32)])
    first = strategy.configure_fit(1, parameters, manager)
    assert len(first) == len({c.cid for c, _ in first}) == 4
    assert strategy._reserved_state_by_round[1] == 0
    strategy._known_state_proxy_ids = {'0', '1', 'absent-client'}
    second = strategy.configure_fit(2, parameters, manager)
    selected = {c.cid for c, _ in second}
    assert len(second) == len(selected) == 4
    assert {'0', '1'} <= selected
    assert 'absent-client' not in selected
    assert strategy._reserved_state_by_round[2] == 2


def test_resume_matches_shipped_dataset_metadata():
    from src.data.behavior_loader import BehaviorDataLoader
    loader = BehaviorDataLoader(str(ROOT / 'data/processed/behaviorfl_dataset.pt'))
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    for run in manifest['runs']:
        assert _load_existing_result(ROOT / 'results', run['method'], run['seed'],
                                     60, 0.02, loader.metadata, 5) is not None, run
