"""
VED (Vehicle Energy Dataset) behavior extractor.

Produces 15-dim behavior vectors from 383 vehicles, Ann Arbor MI.
Tier 1 (10 features): GPS-derived speed/accel/braking dynamics.
Tier 2 (5 features): OBD-II (RPM, fuel rate, engine load, MAF variability proxy).

Dataset: data/raw/ved/Data/VED_DynamicData_Part{1,2}/*.csv
Columns: DayNum, VehId, Trip, Timestamp(ms), Latitude, Longitude,
         Vehicle Speed[km/h], MAF[g/sec], Engine RPM[RPM], Absolute Load[%],
         OAT[DegC], Fuel Rate[L/hr], AC Power, HV Battery*, Fuel Trim*
"""

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "src.data.behavior_extractors"

from ..paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Column names in VED CSVs
COL_VEH_ID = "VehId"
COL_TRIP = "Trip"
COL_TS = "Timestamp(ms)"
COL_LAT = "Latitude[deg]"
COL_LON = "Longitude[deg]"
COL_SPEED = "Vehicle Speed[km/h]"
COL_MAF = "MAF[g/sec]"
COL_RPM = "Engine RPM[RPM]"
COL_LOAD = "Absolute Load[%]"
COL_FUEL = "Fuel Rate[L/hr]"

# Tier 1 feature indices (0-9), Tier 2 feature indices (10-14)
TIER1_FEATURES = [
    "mean_speed", "std_speed", "max_speed",
    "mean_accel", "std_accel",
    "harsh_brake_count", "harsh_accel_count",
    "mean_jerk", "stop_fraction", "speed_entropy",
]
TIER2_FEATURES = [
    "mean_rpm", "rpm_variability",
    "mean_fuel_rate", "mean_engine_load",
    "maf_variability_proxy",
]
ALL_FEATURES = TIER1_FEATURES + TIER2_FEATURES  # 15 total

# Fixed, physically interpretable scales.  Using constants instead of statistics
# computed over the complete artifact prevents held-out vehicles from affecting
# feature preprocessing.  Counts and heavy-tailed quantities are clipped before
# division; all returned model inputs are in [0, 1].
FEATURE_SCALES = np.array(
    [80.0, 50.0, 180.0, 5.0, 5.0, 20.0, 20.0, 10.0, 1.0, 3.0,
     4000.0, 1500.0, 10.0, 100.0, 3.0],
    dtype=np.float32,
)

# Thresholds
MIN_POINTS_PER_TRIP = 50        # ~10s at 200ms sampling
HARSH_BRAKE_THRESHOLD = -3.0    # m/s^2
HARSH_ACCEL_THRESHOLD = 2.5     # m/s^2
STOP_SPEED_THRESHOLD = 1.0      # km/h


def _compute_tier1(speed_kmh: np.ndarray, ts_ms: np.ndarray) -> np.ndarray:
    """Compute 10 Tier 1 features from speed array + timestamps.
    Returns feature_vector[10]."""
    speed_ms = speed_kmh / 3.6  # convert to m/s

    # Time deltas in seconds
    dt = np.diff(ts_ms) / 1000.0  # ms → s
    dt = np.maximum(dt, 0.01)  # avoid division by zero

    # Acceleration [m/s^2]
    accel = np.diff(speed_ms) / dt  # [N-1]
    # Jerk [m/s^3]
    if len(accel) > 1:
        jerk = np.diff(accel) / dt[:-1]
        mean_jerk = float(np.mean(np.abs(jerk)))
    else:
        mean_jerk = 0.0

    # Feature computation
    mean_speed = float(np.mean(speed_kmh))
    std_speed = float(np.std(speed_kmh))
    max_speed = float(np.max(speed_kmh))
    mean_accel = float(np.mean(np.abs(accel))) if len(accel) > 0 else 0.0
    std_accel = float(np.std(accel)) if len(accel) > 0 else 0.0
    harsh_brake_count = int(np.sum(accel < HARSH_BRAKE_THRESHOLD))
    harsh_accel_count = int(np.sum(accel > HARSH_ACCEL_THRESHOLD))
    stop_fraction = float(np.mean(speed_kmh < STOP_SPEED_THRESHOLD))

    # Speed entropy: discretise speed into 20 bins [0, max+1]
    max_v = max(float(np.max(speed_kmh)), 1.0)
    bins = np.linspace(0, max_v + 1, 21)
    hist, _ = np.histogram(speed_kmh, bins=bins)
    hist = hist / (hist.sum() + 1e-8)
    speed_ent = float(scipy_entropy(hist + 1e-8))

    features = np.array([
        mean_speed, std_speed, max_speed,
        mean_accel, std_accel,
        float(harsh_brake_count), float(harsh_accel_count),
        mean_jerk, stop_fraction, speed_ent,
    ], dtype=np.float32)
    return features


def _compute_tier2(
    rpm: np.ndarray, fuel: np.ndarray, load: np.ndarray, maf: np.ndarray
) -> tuple:
    """Compute 5 Tier 2 features + per-feature mask (1.0 if available, 0.0 if all NaN).
    Returns (feature_vector[5], mask[5])."""
    mask = np.zeros(5, dtype=np.float32)
    feats = np.zeros(5, dtype=np.float32)

    # mean_rpm
    valid_rpm = rpm[~np.isnan(rpm)]
    if len(valid_rpm) > 0:
        feats[0] = float(np.mean(valid_rpm))
        feats[1] = float(np.std(valid_rpm))
        mask[0] = mask[1] = 1.0

    # mean_fuel_rate
    valid_fuel = fuel[~np.isnan(fuel)]
    if len(valid_fuel) > 0:
        feats[2] = float(np.mean(valid_fuel))
        mask[2] = 1.0

    # mean_engine_load
    valid_load = load[~np.isnan(load)]
    if len(valid_load) > 0:
        feats[3] = float(np.mean(valid_load))
        mask[3] = 1.0

    # MAF variability proxy: std(MAF) / (mean(MAF) + eps)
    valid_maf = maf[~np.isnan(maf)]
    if len(valid_maf) > 5:
        feats[4] = float(np.std(valid_maf) / (np.mean(valid_maf) + 1e-6))
        mask[4] = 1.0

    return feats, mask


def normalize_behavior_features(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the fixed public feature scaling used by all three sources."""
    scaled = np.clip(features / FEATURE_SCALES, 0.0, 1.0)
    return (scaled * (mask > 0).astype(np.float32)).astype(np.float32)


class VEDExtractor:
    """Extract 15-dim behavior vectors from VED dataset.

    Usage:
        extractor = VEDExtractor(data_root="data/raw/ved")
        result = extractor.extract_all()
        torch.save(result, "data/processed/ved_behaviors.pt")
    """

    def __init__(self, data_root: str | Path = RAW_DATA_DIR / "ved"):
        self.data_root = data_root
        self.part1_dir = os.path.join(data_root, "Data", "VED_DynamicData_Part1")
        self.part2_dir = os.path.join(data_root, "Data", "VED_DynamicData_Part2")

    def _list_csvs(self) -> list[str]:
        csvs = sorted(glob.glob(os.path.join(self.part1_dir, "**", "*.csv"), recursive=True))
        csvs += sorted(glob.glob(os.path.join(self.part2_dir, "**", "*.csv"), recursive=True))
        return csvs

    def extract_all(self, prefix_fraction: float = 0.25) -> dict:
        """Extract behavior features from all VED CSVs.

        Returns dict with keys:
          - trip_features: Tensor [N, 15]
          - tier_mask: Tensor [N, 15]  (1.0 = available, 0.0 = missing)
          - vehicle_ids: Tensor [N]
          - trip_ids: Tensor [N]
          - trip_features: Tensor [N, 15] from the observed trip prefix
          - full_trip_features_raw: Tensor [N, 15] for target construction only
          - start_coords/end_coords: Tensor [N, 2] using valid GPS endpoints
          - state_components_raw: Tensor [N, 3] (RPM, load, MAF variability)
          - source_dataset: List[str]  (all 'ved')
        """
        csvs = self._list_csvs()
        print(f"[VEDExtractor] Found {len(csvs)} CSV files")

        all_prefix_features, all_prefix_masks, all_full_features = [], [], []
        all_full_masks, all_veh_ids, all_trip_ids = [], [], []
        all_starts, all_prefix_ends, all_ends = [], [], []
        all_start_times, all_trip_keys = [], []

        for file_index, csv_path in enumerate(tqdm(csvs, desc="Processing VED CSVs")):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"  [WARN] Failed to read {csv_path}: {e}")
                continue

            # Ensure required columns exist
            required = [COL_VEH_ID, COL_TRIP, COL_TS, COL_SPEED]
            if not all(c in df.columns for c in required):
                print(f"  [WARN] Missing columns in {csv_path}")
                continue

            # Process each (VehId, Trip) group
            for (veh_id, trip_id), group in df.groupby([COL_VEH_ID, COL_TRIP]):
                group = group.sort_values(COL_TS)

                if len(group) < MIN_POINTS_PER_TRIP:
                    continue

                speed = group[COL_SPEED].values.astype(np.float32)
                ts = group[COL_TS].values.astype(np.float64)

                # Skip trips with bad speed data
                if np.any(np.isnan(speed)) or np.any(speed < 0):
                    speed = np.where(np.isnan(speed) | (speed < 0), 0.0, speed)

                # Tier 1 features
                t1 = _compute_tier1(speed, ts)

                # Tier 2 features
                rpm = group[COL_RPM].values.astype(np.float32) if COL_RPM in group.columns else np.full(len(group), np.nan)
                fuel = group[COL_FUEL].values.astype(np.float32) if COL_FUEL in group.columns else np.full(len(group), np.nan)
                load = group[COL_LOAD].values.astype(np.float32) if COL_LOAD in group.columns else np.full(len(group), np.nan)
                maf = group[COL_MAF].values.astype(np.float32) if COL_MAF in group.columns else np.full(len(group), np.nan)
                t2, t2_mask = _compute_tier2(rpm, fuel, load, maf)

                # Predict from an early, deployment-available prefix.  The full
                # trip is retained only for constructing the vehicle-state target.
                prefix_n = min(len(group), max(12, int(np.ceil(prefix_fraction * len(group)))))
                prefix = group.iloc[:prefix_n]
                p_speed = prefix[COL_SPEED].values.astype(np.float32)
                p_speed = np.where(np.isnan(p_speed) | (p_speed < 0), 0.0, p_speed)
                p_ts = prefix[COL_TS].values.astype(np.float64)
                p_t1 = _compute_tier1(p_speed, p_ts)

                def _prefix_values(column: str) -> np.ndarray:
                    return (prefix[column].values.astype(np.float32)
                            if column in prefix.columns
                            else np.full(len(prefix), np.nan, dtype=np.float32))

                p_t2, p_t2_mask = _compute_tier2(
                    _prefix_values(COL_RPM), _prefix_values(COL_FUEL),
                    _prefix_values(COL_LOAD), _prefix_values(COL_MAF),
                )

                # Combine
                full_features = np.concatenate([t1, t2]).astype(np.float32)
                prefix_features = np.concatenate([p_t1, p_t2]).astype(np.float32)
                # Tier 1 mask: all 1.0 (always available from GPS/speed)
                t1_mask = np.ones(10, dtype=np.float32)
                full_mask = np.concatenate([t1_mask, t2_mask]).astype(np.float32)
                prefix_mask = np.concatenate([t1_mask, p_t2_mask]).astype(np.float32)

                # True endpoints, not the mean coordinate used by the legacy
                # recommendation artifact.
                if COL_LAT in group.columns and COL_LON in group.columns:
                    valid = group[[COL_LAT, COL_LON]].dropna()
                    prefix_valid = prefix[[COL_LAT, COL_LON]].dropna()
                    if len(valid):
                        start = valid.iloc[0].to_numpy(dtype=np.float32)
                        end = valid.iloc[-1].to_numpy(dtype=np.float32)
                        prefix_end = (
                            prefix_valid.iloc[-1].to_numpy(dtype=np.float32)
                            if len(prefix_valid) else start.copy()
                        )
                    else:
                        start = prefix_end = end = np.array(
                            [np.nan, np.nan], dtype=np.float32
                        )
                else:
                    start = prefix_end = end = np.array(
                        [np.nan, np.nan], dtype=np.float32
                    )

                day = float(group["DayNum"].iloc[0]) if "DayNum" in group.columns else 0.0
                all_prefix_features.append(prefix_features)
                all_prefix_masks.append(prefix_mask)
                all_full_features.append(full_features)
                all_full_masks.append(full_mask)
                all_veh_ids.append(int(veh_id))
                all_trip_ids.append(len(all_trip_ids))
                all_starts.append(start)
                all_prefix_ends.append(prefix_end)
                all_ends.append(end)
                all_start_times.append(day * 86400.0 + float(ts[0]) / 1000.0)
                all_trip_keys.append(f"ved:{file_index}:{veh_id}:{trip_id}")

        if not all_prefix_features:
            raise RuntimeError("No trips extracted from VED data")

        prefix_raw = np.stack(all_prefix_features)
        prefix_mask = np.stack(all_prefix_masks)
        full_raw = np.stack(all_full_features)
        full_mask = np.stack(all_full_masks)
        feat_norm = normalize_behavior_features(prefix_raw, prefix_mask)

        print(f"[VEDExtractor] Extracted {len(all_prefix_features)} trips from "
              f"{len(set(all_veh_ids))} vehicles")

        return {
            "trip_features": torch.tensor(feat_norm, dtype=torch.float32),
            "tier_mask": torch.tensor(prefix_mask, dtype=torch.float32),
            "prefix_features_raw": torch.tensor(prefix_raw, dtype=torch.float32),
            "full_trip_features_raw": torch.tensor(full_raw, dtype=torch.float32),
            "full_tier_mask": torch.tensor(full_mask, dtype=torch.float32),
            "state_components_raw": torch.tensor(full_raw[:, [10, 13, 14]], dtype=torch.float32),
            "state_components_mask": torch.tensor(full_mask[:, [10, 13, 14]], dtype=torch.float32),
            "vehicle_ids": torch.tensor(all_veh_ids, dtype=torch.long),
            "trip_ids": torch.tensor(all_trip_ids, dtype=torch.long),
            "trip_keys": all_trip_keys,
            "start_times": torch.tensor(all_start_times, dtype=torch.float64),
            "start_coords": torch.tensor(np.stack(all_starts), dtype=torch.float32),
            "prefix_end_coords": torch.tensor(
                np.stack(all_prefix_ends), dtype=torch.float32
            ),
            "end_coords": torch.tensor(np.stack(all_ends), dtype=torch.float32),
            "gps_coords": torch.tensor(np.stack(all_ends), dtype=torch.float32),
            "source_dataset": ["ved"] * len(all_prefix_features),
            "feature_names": ALL_FEATURES,
            "normalization": {"type": "fixed_physical_scales", "scales": FEATURE_SCALES.tolist()},
            "prefix_fraction": float(prefix_fraction),
        }


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else RAW_DATA_DIR / "ved"
    extractor = VEDExtractor(data_root=data_root)
    result = extractor.extract_all()
    out_path = PROCESSED_DATA_DIR / "ved_behaviors.pt"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"Saved to {out_path}")
    print(f"  trip_features shape: {result['trip_features'].shape}")
    print(f"  tier_mask shape: {result['tier_mask'].shape}")
    print(f"  vehicles: {result['vehicle_ids'].unique().numel()}")
    t2_coverage = (result['tier_mask'][:, 10:].sum(dim=1) > 0).float().mean().item()
    print(f"  Tier 2 coverage: {t2_coverage:.1%}")
