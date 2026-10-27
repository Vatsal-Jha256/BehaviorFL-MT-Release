"""Shared data locations and raw-source checks."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


def check_raw_data(raw_data_dir: str | Path = RAW_DATA_DIR,
                   sources: tuple[str, ...] = ('ved', 'geolife', 'tdrive')) -> dict[str, int]:
    """Count source files and report missing inputs before extraction starts."""
    root = Path(raw_data_dir).expanduser().resolve()
    counts = {}
    missing = []
    if 'ved' in sources:
        for part in (1, 2):
            directory = root / 'ved' / 'Data' / f'VED_DynamicData_Part{part}'
            count = sum(1 for p in directory.rglob('*.csv') if p.is_file())
            counts[f'ved_part{part}_csvs'] = count
            if not count:
                missing.append(f'{directory}/**/*.csv')
    if 'geolife' in sources:
        directory = root / 'geolife' / 'Data'
        count = sum(1 for p in directory.glob('*/labels.txt')
                    if p.is_file() and any((p.parent / 'Trajectory').glob('*.plt')))
        counts['geolife_labeled_users'] = count
        if not count:
            missing.append(f'{directory}/<user>/labels.txt and Trajectory/*.plt')
    if 'tdrive' in sources:
        directory = root / 'tdrive' / 'release' / 'taxi_log_2008_by_id'
        count = sum(1 for p in directory.glob('*.txt') if p.is_file())
        counts['tdrive_vehicle_files'] = count
        if not count:
            missing.append(f'{directory}/*.txt')
    if missing:
        raise FileNotFoundError('Missing raw data:\n  ' + '\n  '.join(missing)
                                + '\nSee data/README.md for download and extraction instructions.')
    return counts
