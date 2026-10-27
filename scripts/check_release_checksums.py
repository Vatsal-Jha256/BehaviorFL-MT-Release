#!/usr/bin/env python3
"""Check the dataset, saved results, and checkpoints."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
errors = []
for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    path = ROOT / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name)
for name in errors:
    print(f'[checksums] FAIL {name}')
print(f"[checksums] {'FAIL' if errors else 'PASS'}")
raise SystemExit(bool(errors))
