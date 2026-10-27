"""Raw-data layout and preprocessing entrypoint checks."""
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.paths import check_raw_data
from src.data.behavior_extractors.ved_extractor import VEDExtractor
from scripts.behavior import assemble_behaviorfl_dataset as assembly


@pytest.fixture
def raw_layout(tmp_path):
    root = tmp_path / 'raw inputs'
    # Some archive tools preserve a directory inside each extracted VED part.
    paths = ['ved/Data/VED_DynamicData_Part1/nested/week1.csv',
             'ved/Data/VED_DynamicData_Part2/week2.csv',
             'geolife/Data/000/labels.txt',
             'geolife/Data/000/Trajectory/trip.plt',
             'tdrive/release/taxi_log_2008_by_id/1.txt']
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('layout fixture\n')
    return root


def test_raw_discovery_supports_extracted_archive_directories(raw_layout):
    assert check_raw_data(raw_layout) == {'ved_part1_csvs': 1, 'ved_part2_csvs': 1,
                                         'geolife_labeled_users': 1, 'tdrive_vehicle_files': 1}
    csvs = VEDExtractor(raw_layout / 'ved')._list_csvs()
    assert len(csvs) == 2
    assert {Path(p).name for p in csvs} == {'week1.csv', 'week2.csv'}


def test_missing_sources_fail_before_any_processed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, 'ROOT', tmp_path / 'project')
    with pytest.raises(FileNotFoundError) as error:
        assembly.assemble(raw_data_dir=tmp_path / 'missing', rebuild_extractors=True)
    message = str(error.value)
    assert all(source in message for source in ['ved', 'geolife', 'tdrive'])
    assert 'data/README.md' in message
    assert not (assembly.ROOT / 'data/processed').exists()


def test_raw_check_cli_works_outside_repository(raw_layout, tmp_path):
    script = Path(assembly.__file__).resolve()
    process = subprocess.run([sys.executable, str(script), '--raw-data-dir', str(raw_layout),
                              '--check-raw-data'], cwd=tmp_path, text=True,
                             capture_output=True)
    assert process.returncode == 0, process.stderr
    assert 'ved_part1_csvs=1' in process.stdout
    assert 'geolife_labeled_users=1' in process.stdout
    assert 'tdrive_vehicle_files=1' in process.stdout
    assert not list(tmp_path.rglob('*.pt'))
