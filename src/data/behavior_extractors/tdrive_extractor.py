"""
T-Drive taxi trajectory extractor.

10,357 taxi drivers, Beijing. Sparse GPS (~10 min intervals).
Tier 1 only — coarse features only (4 reliable features, 6 low-quality).
Fine-grained features (accel, jerk) set to 0 with tier_mask=0.5 (partial).

Dataset: data/raw/tdrive/release/taxi_log_2008_by_id/{id}.txt
Format (headerless): driver_id, timestamp, lon, lat
Timestamp format: YYYY-MM-DD HH:MM:SS

Used primarily for scalability testing (10K clients).
"""

import os
from pathlib import Path
import glob
import math
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm
from datetime import datetime, timezone

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "src.data.behavior_extractors"

from ..paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

from .ved_extractor import FEATURE_SCALES, normalize_behavior_features

MIN_POINTS_PER_TRAJECTORY = 10  # sparse data, lower threshold
STOP_SPEED_THRESHOLD = 5.0      # km/h (higher threshold for sparse data)
SEGMENT_GAP_SECONDS = 30 * 60
MAX_SEGMENTS_PER_DRIVER = 20

FEATURE_NAMES = [
    "mean_speed", "std_speed", "max_speed",
    "mean_accel", "std_accel",
    "harsh_brake_count", "harsh_accel_count",
    "mean_jerk", "stop_fraction", "speed_entropy",
    "mean_rpm", "rpm_variability", "mean_fuel_rate", "mean_engine_load", "maf_variability_proxy",
]

# For sparse GPS, only 4 Tier 1 features are reliable
# Indices: 0=mean_speed, 1=std_speed, 2=max_speed, 8=stop_fraction, 9=speed_entropy
RELIABLE_TIER1 = {0, 1, 2, 8, 9}   # mask=1.0
UNRELIABLE_TIER1 = {3, 4, 5, 6, 7}  # mask=0.5 (low-confidence due to sparse sampling)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_tdrive_file(filepath: str) -> "tuple | None":
    """Parse T-Drive file. Returns (lats, lons, timestamps) or None."""
    lats, lons, times = [], [], []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts = datetime.strptime(
                    parts[1].strip(), "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc).timestamp()
                lon = float(parts[2])
                lat = float(parts[3])
                # Basic Beijing bounds filter (approx: lat 39.4-41.0, lon 115.7-117.4)
                if not (39.4 <= lat <= 41.0 and 115.7 <= lon <= 117.4):
                    continue
                lats.append(lat)
                lons.append(lon)
                times.append(ts)
            except (ValueError, IndexError):
                continue

    if len(lats) < MIN_POINTS_PER_TRAJECTORY:
        return None
    return np.array(lats, np.float64), np.array(lons, np.float64), np.array(times, np.float64)


def _trajectory_segments(times: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open daily/inactivity-gap segments for one taxi trace."""
    if len(times) == 0:
        return []
    order = np.argsort(times, kind="stable")
    if not np.array_equal(order, np.arange(len(times))):
        raise ValueError("T-Drive timestamps must be sorted before segmentation")
    day = np.floor(times / 86400.0).astype(np.int64)
    boundaries = np.flatnonzero(
        (np.diff(day) != 0) | (np.diff(times) > SEGMENT_GAP_SECONDS)
    ) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(times)]))
    return [
        (int(a), int(b)) for a, b in zip(starts, stops)
        if int(b - a) >= MIN_POINTS_PER_TRAJECTORY
    ]


def _extract_sparse_features(
    lats: np.ndarray, lons: np.ndarray, times: np.ndarray
) -> "np.ndarray | None":
    """Extract behavior features from sparse GPS trajectory."""
    n = len(lats)
    speeds_kmh = []
    for i in range(1, n):
        dist_km = _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
        dt_h = max((times[i] - times[i-1]) / 3600.0, 1e-6)
        speed = dist_km / dt_h  # km/h
        speeds_kmh.append(min(speed, 200.0))  # cap outliers

    speeds = np.array(speeds_kmh, dtype=np.float32)
    if len(speeds) < 2:
        return None

    # Reliable features
    mean_speed = float(np.mean(speeds))
    std_speed = float(np.std(speeds))
    max_speed = float(np.max(speeds))
    stop_frac = float(np.mean(speeds < STOP_SPEED_THRESHOLD))

    max_v = max(float(np.max(speeds)), 1.0)
    bins = np.linspace(0, max_v + 1, 21)
    hist, _ = np.histogram(speeds, bins=bins)
    hist = hist / (hist.sum() + 1e-8)
    speed_ent = float(scipy_entropy(hist + 1e-8))

    # Unreliable features (sparse → noisy accel/jerk)
    dt_s = np.diff(times[1:])
    dt_s = np.maximum(dt_s, 1.0)
    accel = np.diff(speeds / 3.6) / dt_s
    mean_accel = float(np.mean(np.abs(accel))) if len(accel) > 0 else 0.0
    std_accel = float(np.std(accel)) if len(accel) > 0 else 0.0

    if len(accel) > 1:
        mean_jerk = float(np.mean(np.abs(np.diff(accel) / np.maximum(dt_s[:-1], 1.0))))
    else:
        mean_jerk = 0.0

    t1 = np.array([
        mean_speed, std_speed, max_speed,
        mean_accel, std_accel,
        0.0, 0.0,         # harsh brake/accel: too noisy at sparse sampling
        mean_jerk,
        stop_frac, speed_ent,
    ], dtype=np.float32)
    t2 = np.zeros(5, dtype=np.float32)
    return np.concatenate([t1, t2])


class TDriveExtractor:
    """Extract coarse Tier 1 behavior features from T-Drive Beijing taxis.

    Uses only features reliable at ~10-min GPS sampling intervals.
    Fine-grained features (accel, jerk) get tier_mask=0.5 (low-confidence).

    Usage:
        extractor = TDriveExtractor(data_root="data/raw/tdrive")
        result = extractor.extract_all(max_drivers=500)  # subsample for FL training
        torch.save(result, "data/processed/tdrive_behaviors.pt")
    """

    def __init__(self, data_root: str | Path = RAW_DATA_DIR / "tdrive"):
        self.traj_dir = os.path.join(data_root, "release", "taxi_log_2008_by_id")

    def extract_all(
        self,
        max_drivers: int = 10357,
        prefix_fraction: float = 0.25,
        max_segments_per_driver: int = MAX_SEGMENTS_PER_DRIVER,
    ) -> dict:
        """Extract features from T-Drive taxis.

        Args:
            max_drivers: Max number of drivers to process (default: all 10,357).
                         Set to 500 for FL training subset, 10357 for scalability test.
        """
        files = sorted(glob.glob(os.path.join(self.traj_dir, "*.txt")))[:max_drivers]
        print(f"[TDriveExtractor] Processing {len(files)} driver files")

        all_features, all_masks = [], []
        all_driver_ids, all_trip_ids = [], []
        all_starts, all_prefix_ends, all_ends = [], [], []
        all_start_times, all_trip_keys = [], []

        for fpath in tqdm(files, desc="Processing T-Drive"):
            parsed = _parse_tdrive_file(fpath)
            if parsed is None:
                continue
            lats, lons, times = parsed
            order = np.argsort(times, kind="stable")
            lats, lons, times = lats[order], lons[order], times[order]
            driver_id = int(os.path.basename(fpath).replace(".txt", ""))
            segments = _trajectory_segments(times)[:max_segments_per_driver]
            for trip_id, (start, stop) in enumerate(segments):
                seg_lats, seg_lons, seg_times = lats[start:stop], lons[start:stop], times[start:stop]
                prefix_n = min(
                    len(seg_times), max(4, int(np.ceil(prefix_fraction * len(seg_times))))
                )
                features = _extract_sparse_features(
                    seg_lats[:prefix_n], seg_lons[:prefix_n], seg_times[:prefix_n]
                )
                if features is None:
                    continue
                mask = np.zeros(15, dtype=np.float32)
                for i in RELIABLE_TIER1:
                    mask[i] = 1.0
                # Sparse-sampling dynamics are not treated as fractional
                # observations: they remain explicit low-confidence inputs.
                for i in UNRELIABLE_TIER1:
                    mask[i] = 0.5
                all_features.append(features)
                all_masks.append(mask)
                all_driver_ids.append(driver_id)
                all_trip_ids.append(trip_id)
                all_starts.append(np.array([seg_lats[0], seg_lons[0]], dtype=np.float32))
                all_prefix_ends.append(np.array([
                    seg_lats[prefix_n - 1], seg_lons[prefix_n - 1]
                ], dtype=np.float32))
                all_ends.append(np.array([seg_lats[-1], seg_lons[-1]], dtype=np.float32))
                all_start_times.append(float(seg_times[0]))
                all_trip_keys.append(f"tdrive:{driver_id}:{trip_id}")

        if not all_features:
            raise RuntimeError("No valid trajectories found in T-Drive data")

        feat_array = np.stack(all_features)
        mask_array = np.stack(all_masks)

        feat_norm = normalize_behavior_features(feat_array, mask_array)

        print(
            f"[TDriveExtractor] Extracted {len(all_features)} trajectory segments "
            f"from {len(set(all_driver_ids))} drivers"
        )

        return {
            "trip_features": torch.tensor(feat_norm, dtype=torch.float32),
            "tier_mask": torch.tensor(mask_array, dtype=torch.float32),
            "vehicle_ids": torch.tensor(all_driver_ids, dtype=torch.long),
            "trip_ids": torch.tensor(all_trip_ids, dtype=torch.long),
            "trip_keys": all_trip_keys,
            "start_times": torch.tensor(all_start_times, dtype=torch.float64),
            "start_coords": torch.tensor(np.stack(all_starts), dtype=torch.float32),
            "prefix_end_coords": torch.tensor(
                np.stack(all_prefix_ends), dtype=torch.float32
            ),
            "end_coords": torch.tensor(np.stack(all_ends), dtype=torch.float32),
            "gps_coords": torch.tensor(np.stack(all_ends), dtype=torch.float32),
            "state_components_raw": torch.zeros((len(all_features), 3), dtype=torch.float32),
            "state_components_mask": torch.zeros((len(all_features), 3), dtype=torch.float32),
            "source_dataset": ["tdrive"] * len(all_features),
            "feature_names": FEATURE_NAMES,
            "normalization": {"type": "fixed_physical_scales", "scales": FEATURE_SCALES.tolist()},
            "prefix_fraction": float(prefix_fraction),
            "segmentation": {
                "gap_seconds": SEGMENT_GAP_SECONDS,
                "calendar_day_boundary": True,
                "max_segments_per_driver": int(max_segments_per_driver),
            },
        }


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else RAW_DATA_DIR / "tdrive"
    max_drivers = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    extractor = TDriveExtractor(data_root=data_root)
    result = extractor.extract_all(max_drivers=max_drivers)
    out_path = PROCESSED_DATA_DIR / "tdrive_behaviors.pt"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"Saved to {out_path}")
    print(f"  trip_features shape: {result['trip_features'].shape}")
    reliable_mask = (result['tier_mask'][:, 0] == 1.0).float().mean().item()
    print(f"  Reliable Tier 1 coverage: {reliable_mask:.1%}")
