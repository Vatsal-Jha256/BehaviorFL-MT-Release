"""
GeoLife trajectory extractor.

182 users, Beijing. Dense GPS (1-5 second intervals).
Tier 1 only (10 GPS-derived features). Tier 2 mask = 0.

Dataset: data/raw/geolife/Data/{000-181}/Trajectory/*.plt
PLT format: lat, lon, 0, altitude_ft, days_since_1899, date, time (skip 6 header lines)
Labels: Data/{user}/labels.txt — tab-separated: Start Time, End Time, Transportation Mode
"""

import os
from pathlib import Path
import glob
import math
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm
from datetime import datetime

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "src.data.behavior_extractors"

from ..paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

from .ved_extractor import FEATURE_SCALES, normalize_behavior_features

# Target transport modes (car/taxi only)
CAR_MODES = {"car", "taxi"}

# Same thresholds as VED
HARSH_BRAKE_THRESHOLD = -3.0   # m/s^2
HARSH_ACCEL_THRESHOLD = 2.5    # m/s^2
STOP_SPEED_THRESHOLD = 1.0     # km/h
MIN_POINTS_PER_TRIP = 30       # lower threshold for GPS (sparser)

FEATURE_NAMES = [
    "mean_speed", "std_speed", "max_speed",
    "mean_accel", "std_accel",
    "harsh_brake_count", "harsh_accel_count",
    "mean_jerk", "stop_fraction", "speed_entropy",
    "mean_rpm", "rpm_variability", "mean_fuel_rate", "mean_engine_load", "maf_variability_proxy",
]


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_plt(plt_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse PLT file. Returns (lats, lons, timestamps_seconds)."""
    lats, lons, times = [], [], []
    with open(plt_path, "r") as f:
        for i, line in enumerate(f):
            if i < 6:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                date_str = parts[5].strip()
                time_str = parts[6].strip()
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                ts = dt.timestamp()
                lats.append(lat)
                lons.append(lon)
                times.append(ts)
            except (ValueError, IndexError):
                continue
    return np.array(lats, dtype=np.float64), np.array(lons, dtype=np.float64), np.array(times, dtype=np.float64)


def _parse_labels(label_path: str) -> list[tuple[datetime, datetime, str]]:
    """Parse labels.txt. Returns list of (start, end, mode)."""
    labels = []
    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                start = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S")
                end = datetime.strptime(parts[1].strip(), "%Y/%m/%d %H:%M:%S")
                mode = parts[2].strip().lower()
                labels.append((start, end, mode))
            except ValueError:
                continue
    return labels


def _extract_features_from_trajectory(
    lats: np.ndarray, lons: np.ndarray, times: np.ndarray
) -> "np.ndarray | None":
    """Compute 10 Tier 1 features from GPS trajectory."""
    n = len(lats)
    # Compute speeds (km/h) from consecutive points
    speeds = []
    for i in range(1, n):
        dist_m = _haversine_m(lats[i-1], lons[i-1], lats[i], lons[i])
        dt = max(times[i] - times[i-1], 0.5)  # at least 0.5s
        speed_ms = dist_m / dt
        speeds.append(speed_ms * 3.6)  # m/s → km/h
    speeds = np.array(speeds, dtype=np.float32)
    speeds = np.clip(speeds, 0, 200)  # cap at 200 km/h (noise filter)

    if len(speeds) < 2:
        return None

    # Acceleration from speed
    dt_arr = np.diff(times[1:])  # time between speed samples
    dt_arr = np.maximum(dt_arr, 0.5)
    accel = np.diff(speeds / 3.6) / dt_arr  # m/s^2

    # Jerk
    if len(accel) > 1:
        mean_jerk = float(np.mean(np.abs(np.diff(accel) / np.maximum(dt_arr[:-1], 0.5))))
    else:
        mean_jerk = 0.0

    mean_speed = float(np.mean(speeds))
    std_speed = float(np.std(speeds))
    max_speed = float(np.max(speeds))
    mean_accel = float(np.mean(np.abs(accel))) if len(accel) > 0 else 0.0
    std_accel = float(np.std(accel)) if len(accel) > 0 else 0.0
    harsh_brake = int(np.sum(accel < HARSH_BRAKE_THRESHOLD))
    harsh_accel = int(np.sum(accel > HARSH_ACCEL_THRESHOLD))
    stop_frac = float(np.mean(speeds < STOP_SPEED_THRESHOLD))

    max_v = max(float(np.max(speeds)), 1.0)
    bins = np.linspace(0, max_v + 1, 21)
    hist, _ = np.histogram(speeds, bins=bins)
    hist = hist / (hist.sum() + 1e-8)
    speed_ent = float(scipy_entropy(hist + 1e-8))

    t1 = np.array([
        mean_speed, std_speed, max_speed,
        mean_accel, std_accel,
        float(harsh_brake), float(harsh_accel),
        mean_jerk, stop_frac, speed_ent,
    ], dtype=np.float32)
    t2 = np.zeros(5, dtype=np.float32)  # no OBD data
    return np.concatenate([t1, t2])


class GeoLifeExtractor:
    """Extract Tier 1 behavior features from GeoLife dataset.

    Only uses users with car/taxi mode labels (69 labeled users).
    Tier 2 features are zero with mask=0.

    Usage:
        extractor = GeoLifeExtractor(data_root="data/raw/geolife")
        result = extractor.extract_all()
        torch.save(result, "data/processed/geolife_behaviors.pt")
    """

    def __init__(self, data_root: str | Path = RAW_DATA_DIR / "geolife"):
        self.data_root = data_root
        self.data_dir = os.path.join(data_root, "Data")

    def _get_labeled_users(self) -> dict[str, list[tuple]]:
        """Returns {user_id: [(start_ts, end_ts, mode), ...]} for users with labels."""
        user_labels = {}
        user_dirs = sorted(os.listdir(self.data_dir))
        for user_id in user_dirs:
            label_path = os.path.join(self.data_dir, user_id, "labels.txt")
            if os.path.exists(label_path):
                labels = _parse_labels(label_path)
                car_labels = [(l[0].timestamp(), l[1].timestamp(), l[2])
                              for l in labels if l[2] in CAR_MODES]
                if car_labels:
                    user_labels[user_id] = car_labels
        return user_labels

    def extract_all(self, prefix_fraction: float = 0.25) -> dict:
        """Extract behavior features for car/taxi trips from labeled GeoLife users.

        Returns dict with same schema as VEDExtractor.extract_all().
        """
        user_labels = self._get_labeled_users()
        print(f"[GeoLifeExtractor] Found {len(user_labels)} users with car/taxi labels")

        all_features, all_masks = [], []
        all_user_ids, all_trip_ids = [], []
        all_starts, all_prefix_ends, all_ends = [], [], []
        all_start_times, all_trip_keys = [], []

        for user_id, car_intervals in tqdm(user_labels.items(), desc="Processing GeoLife users"):
            traj_dir = os.path.join(self.data_dir, user_id, "Trajectory")
            if not os.path.exists(traj_dir):
                continue

            plt_files = sorted(glob.glob(os.path.join(traj_dir, "*.plt")))
            trip_counter = 0

            for plt_path in plt_files:
                try:
                    lats, lons, times = _parse_plt(plt_path)
                except Exception:
                    continue

                if len(times) < MIN_POINTS_PER_TRIP:
                    continue

                # Intersect the PLT trace with each real car/taxi label interval.
                # This avoids treating unrelated points elsewhere in the file as
                # part of the driving trip.
                for interval_index, (start_ts, end_ts, _) in enumerate(car_intervals):
                    select = (times >= start_ts) & (times <= end_ts)
                    if int(select.sum()) < MIN_POINTS_PER_TRIP:
                        continue
                    seg_lats, seg_lons, seg_times = lats[select], lons[select], times[select]
                    prefix_n = min(
                        len(seg_times), max(8, int(np.ceil(prefix_fraction * len(seg_times))))
                    )
                    features = _extract_features_from_trajectory(
                        seg_lats[:prefix_n], seg_lons[:prefix_n], seg_times[:prefix_n]
                    )
                    if features is None:
                        continue

                    mask = np.concatenate([
                        np.ones(10, dtype=np.float32), np.zeros(5, dtype=np.float32)
                    ])
                    all_features.append(features)
                    all_masks.append(mask)
                    all_user_ids.append(int(user_id))
                    all_trip_ids.append(trip_counter)
                    all_starts.append(np.array([seg_lats[0], seg_lons[0]], dtype=np.float32))
                    all_prefix_ends.append(np.array([
                        seg_lats[prefix_n - 1], seg_lons[prefix_n - 1]
                    ], dtype=np.float32))
                    all_ends.append(np.array([seg_lats[-1], seg_lons[-1]], dtype=np.float32))
                    all_start_times.append(float(seg_times[0]))
                    all_trip_keys.append(
                        f"geolife:{user_id}:{os.path.basename(plt_path)}:{interval_index}"
                    )
                    trip_counter += 1

        if not all_features:
            raise RuntimeError("No car/taxi trips found in GeoLife data")

        feat_array = np.stack(all_features)
        mask_array = np.stack(all_masks)
        feat_norm = normalize_behavior_features(feat_array, mask_array)

        print(f"[GeoLifeExtractor] Extracted {len(all_features)} car/taxi trips "
              f"from {len(user_labels)} users")

        return {
            "trip_features": torch.tensor(feat_norm, dtype=torch.float32),
            "tier_mask": torch.tensor(mask_array, dtype=torch.float32),
            "vehicle_ids": torch.tensor(all_user_ids, dtype=torch.long),
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
            "source_dataset": ["geolife"] * len(all_features),
            "feature_names": FEATURE_NAMES,
            "normalization": {"type": "fixed_physical_scales", "scales": FEATURE_SCALES.tolist()},
            "prefix_fraction": float(prefix_fraction),
        }


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else RAW_DATA_DIR / "geolife"
    extractor = GeoLifeExtractor(data_root=data_root)
    result = extractor.extract_all()
    out_path = PROCESSED_DATA_DIR / "geolife_behaviors.pt"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"Saved to {out_path}")
    print(f"  trip_features shape: {result['trip_features'].shape}")
    print(f"  users: {result['vehicle_ids'].unique().numel()}")
