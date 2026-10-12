"""
Standalone PyTorch data loader for DayCent synthetic data.

The data layout used here is:

    SAS_KGML_090925/
    ├── InputData/
    │   ├── WeatherData/{point_id}.csv
    │   ├── initial_site_conditions.xlsx
    │   ├── sample_points_lookup_SAS.xlsx
    │   └── schedule_scenarios_all_Synthetic_10000.csv
    └── OutputData_Synthetic_10000/
        ├── SAS_scenario_{scenario_id}_harvest.csv
        └── SAS_scenario_{scenario_id}_monthly.csv

"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = "/path/to/daycent/data"


def worker_init(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic numpy seed."""
    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        np.random.seed(worker_info.seed % (2**32 - 1))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _params_to_dict(params: Any) -> Dict[str, Any]:
    if isinstance(params, dict):
        return params
    if hasattr(params, "params"):
        return params.params
    raise TypeError("params must be a dict or YParams-like object with .params")


def _range_or_list(split_cfg: Dict[str, Any], key: str, range_key: str) -> List[int]:
    if key in split_cfg and split_cfg[key] is not None:
        return [int(x) for x in split_cfg[key]]
    if range_key in split_cfg and split_cfg[range_key] is not None:
        start = int(split_cfg[range_key]["start"])
        end = int(split_cfg[range_key]["end"])
        return list(range(start, end + 1))
    return []


@dataclass
class ZScoreNormalizer:
    """Tiny z-score normalizer with train-set NaN imputation."""

    mean: np.ndarray
    std: np.ndarray
    columns: List[str]

    @classmethod
    def fit(
        cls,
        values: pd.DataFrame | np.ndarray,
        columns: Iterable[str],
        min_std: float = 1.0e-6,
    ) -> "ZScoreNormalizer":
        columns = list(columns)
        arr = values[columns].to_numpy(dtype=np.float64) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=np.float64)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where((np.isnan(std)) | (std < min_std), 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), columns=columns)

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        arr = np.where(np.isnan(arr), self.mean, arr)
        return (arr - self.mean) / self.std

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform_array(frame[self.columns].to_numpy(dtype=np.float32))

    def transform_scalar(self, value: float) -> float:
        return float((float(value) - self.mean[0]) / self.std[0])

    def inverse_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return arr * self.std + self.mean

    def inverse_scalar(self, value: float) -> float:
        return float(float(value) * self.std[0] + self.mean[0])


class DayCentYieldDataset(Dataset):
    """
    Yield-focused DayCent Dataset for one split.

    Samples are keyed by ``(scenario_id, year, point_id)`` and returned as a
    dictionary suitable for PyTorch workflows.
    """

    def __init__(self, params: Any, split: str = "train"):
        self.config = _params_to_dict(params)
        self.split = split
        self.data_root = self.config.get("data_root", DEFAULT_DATA_ROOT)
        self.data_cfg = self.config.get("data", {})
        self.splits_cfg = self.config.get("splits", {})
        self.normalization_cfg = self.config.get("normalization", {})

        if split not in self.splits_cfg:
            available = ", ".join(sorted(self.splits_cfg))
            raise KeyError(f"Unknown split {split!r}. Available splits: {available}")

        self.input_dir = os.path.join(self.data_root, "InputData")
        self.weather_dir = os.path.join(self.input_dir, "WeatherData")
        self.output_dir = os.path.join(self.data_root, "OutputData_Synthetic_10000")
        self.lookup_path = os.path.join(self.input_dir, "sample_points_lookup_SAS.xlsx")
        self.initial_conditions_path = os.path.join(self.input_dir, "initial_site_conditions.xlsx")
        self.schedule_path = os.path.join(self.input_dir, "schedule_scenarios_all_Synthetic_10000.csv")

        self.weather_columns = list(self.data_cfg.get("weather_columns", ["Tmax", "Tmin", "Precip"]))
        self.management_events = list(self.data_cfg.get("management_events", []))
        if not self.management_events:
            raise ValueError("config.data.management_events must define a stable event vocabulary")

        self.all_point_ids = self._load_all_point_ids()
        self.scenario_ids, self.point_ids, self.years = self._resolve_split(split)
        self.train_scenario_ids, self.train_point_ids, self.train_years = self._resolve_split(
            self.normalization_cfg.get("fit_split", "train")
        )

        self._validate_paths()
        self._validate_scenario_files(self.scenario_ids)
        self._validate_scenario_files(self.train_scenario_ids)

        self.static_raw, self.static_columns = self._load_initial_conditions()
        self.normalizers = self._fit_train_normalizers()

        self.feature_names = (
            self.weather_columns
            + [f"event_{event}" for event in self.management_events]
            + ["doy_sin", "doy_cos"]
        )
        self.static_feature_names = self.static_columns
        self.doy_features = self._build_doy_features()

        self.weather_by_key = self._load_weather()
        self.management_by_key = self._load_management()
        self.static_by_point = self._build_static_lookup()
        self.output_by_key, self.output_mask_by_key = self._load_outputs()

        self.sample_keys = list(itertools.product(self.scenario_ids, self.years, self.point_ids))

    def __len__(self) -> int:
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        scenario_id, year, point_id = self.sample_keys[idx]
        weather = self.weather_by_key[(point_id, year)]
        management = self.management_by_key[(scenario_id, year)]
        daily_input = np.concatenate([weather, management, self.doy_features], axis=1).astype(np.float32)

        output_key = (scenario_id, point_id, year)
        output = self.output_by_key.get(output_key, 0.0)
        output_mask = self.output_mask_by_key.get(output_key, 0.0)

        return {
            "input": torch.tensor(daily_input, dtype=torch.float32),
            "static": torch.tensor(self.static_by_point[point_id], dtype=torch.float32),
            "output": torch.tensor(output, dtype=torch.float32),
            "output_mask": torch.tensor(output_mask, dtype=torch.float32),
            "scenario_id": scenario_id,
            "point_id": point_id,
            "year": str(year),
            "global_idx": idx,
        }

    def denormalize_weather(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized weather columns back to native data units."""
        return self.normalizers["weather"].inverse_array(values)

    def denormalize_output(self, value: float) -> float:
        """Convert normalized ``cgrain`` back to g C m-2 harvest-1."""
        return self.normalizers["output"].inverse_scalar(value)

    def _load_all_point_ids(self) -> List[str]:
        lookup = pd.read_excel(self.lookup_path, sheet_name="sample_points_lookup_SAS")
        ids = lookup["id"].astype(str).tolist()
        if self.data_cfg.get("point_sort", "numeric") == "numeric":
            return sorted(ids, key=lambda x: int(x))
        return sorted(ids)

    def _resolve_split(self, split: str) -> Tuple[List[str], List[str], List[int]]:
        split_cfg = self.splits_cfg[split]
        scenarios = [str(s) for s in split_cfg.get("scenarios", [])]
        years = _range_or_list(split_cfg, "years", "year_range")

        if "points" in split_cfg and split_cfg["points"] is not None:
            point_ids = [str(p) for p in split_cfg["points"]]
        elif "point_slice" in split_cfg and split_cfg["point_slice"] is not None:
            start = int(split_cfg["point_slice"]["start"])
            end = int(split_cfg["point_slice"]["end"])
            point_ids = self.all_point_ids[start:end]
        else:
            raise ValueError(f"Split {split!r} must define either points or point_slice")

        if not scenarios or not years or not point_ids:
            raise ValueError(
                f"Split {split!r} resolved to scenarios={len(scenarios)}, "
                f"points={len(point_ids)}, years={len(years)}"
            )
        return scenarios, point_ids, years

    def _validate_paths(self) -> None:
        required_paths = [
            self.data_root,
            self.input_dir,
            self.weather_dir,
            self.output_dir,
            self.lookup_path,
            self.initial_conditions_path,
            self.schedule_path,
        ]
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError("Missing required DayCent paths:\n" + "\n".join(missing))

    def _validate_scenario_files(self, scenario_ids: List[str]) -> None:
        missing = []
        validate_monthly = bool(self.data_cfg.get("validate_monthly_files", True))
        for scenario_id in scenario_ids:
            harvest_path = self._harvest_path(scenario_id)
            monthly_path = self._monthly_path(scenario_id)
            if not os.path.exists(harvest_path):
                missing.append(harvest_path)
            if validate_monthly and not os.path.exists(monthly_path):
                missing.append(monthly_path)
        if missing:
            raise FileNotFoundError(
                "Missing DayCent scenario output files. "
                "The old synthetic loader expects SAS_scenario_{id}_harvest.csv "
                "and SAS_scenario_{id}_monthly.csv files:\n"
                + "\n".join(missing)
            )

    def _harvest_path(self, scenario_id: str) -> str:
        return os.path.join(self.output_dir, f"SAS_scenario_{scenario_id}_harvest.csv")

    def _monthly_path(self, scenario_id: str) -> str:
        return os.path.join(self.output_dir, f"SAS_scenario_{scenario_id}_monthly.csv")

    def _load_initial_conditions(self) -> Tuple[pd.DataFrame, List[str]]:
        frame = pd.read_excel(self.initial_conditions_path, sheet_name="initial_conditions")
        frame["id"] = frame["id"].astype(str)
        frame = frame.set_index("id")
        frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame, frame.columns.tolist()

    def _fit_train_normalizers(self) -> Dict[str, ZScoreNormalizer]:
        min_std = float(self.normalization_cfg.get("min_std", 1.0e-6))

        weather_frames = []
        for point_id in self.train_point_ids:
            weather_frames.append(
                self._read_weather_file(point_id, years=self.train_years, normalize=False)
            )
        weather_train = pd.concat(weather_frames, ignore_index=True)
        weather_norm = ZScoreNormalizer.fit(weather_train, self.weather_columns, min_std=min_std)

        static_train = self.static_raw.loc[self.train_point_ids, self.static_columns]
        static_norm = ZScoreNormalizer.fit(static_train, self.static_columns, min_std=min_std)

        output_values = []
        for scenario_id in self.train_scenario_ids:
            harvest = self._read_harvest_file(scenario_id)
            mask = (
                harvest["point_id"].isin(self.train_point_ids)
                & harvest["Year"].isin(self.train_years)
            )
            output_values.append(harvest.loc[mask, ["cgrain"]])
        output_train = pd.concat(output_values, ignore_index=True)
        if output_train["cgrain"].dropna().empty:
            raise ValueError("No training cgrain values found for output normalization")
        output_norm = ZScoreNormalizer.fit(output_train, ["cgrain"], min_std=min_std)

        return {
            "weather": weather_norm,
            "static": static_norm,
            "output": output_norm,
        }

    def _read_weather_file(self, point_id: str, years: List[int], normalize: bool) -> pd.DataFrame | Dict[int, np.ndarray]:
        path = os.path.join(self.weather_dir, f"{point_id}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weather file for point {point_id}: {path}")

        usecols = ["Year", "doy"] + self.weather_columns
        frame = pd.read_csv(path, usecols=usecols)
        frame = frame[frame["Year"].isin(years)]
        frame = frame[(frame["doy"] >= 1) & (frame["doy"] <= 365)]
        frame = frame.sort_values(["Year", "doy"])

        if not normalize:
            return frame

        by_year: Dict[int, np.ndarray] = {}
        for year in years:
            year_frame = frame[frame["Year"] == year].drop_duplicates("doy", keep="last")
            year_frame = year_frame.set_index("doy").reindex(range(1, 366))
            values = year_frame[self.weather_columns].to_numpy(dtype=np.float32)
            by_year[year] = self.normalizers["weather"].transform_array(values).astype(np.float32)
        return by_year

    def _load_weather(self) -> Dict[Tuple[str, int], np.ndarray]:
        weather_by_key: Dict[Tuple[str, int], np.ndarray] = {}
        for point_id in self.point_ids:
            point_weather = self._read_weather_file(point_id, self.years, normalize=True)
            for year, values in point_weather.items():
                weather_by_key[(point_id, year)] = values
        return weather_by_key

    def _load_management(self) -> Dict[Tuple[str, int], np.ndarray]:
        management_by_key = {
            (scenario_id, year): np.zeros((365, len(self.management_events)), dtype=np.float32)
            for scenario_id in self.scenario_ids
            for year in self.years
        }
        event_to_idx = {event: idx for idx, event in enumerate(self.management_events)}

        schedule = pd.read_csv(
            self.schedule_path,
            usecols=["scenario", "simyear", "doy", "management"],
        )
        schedule["scenario_id"] = schedule["scenario"].astype(str).str.replace("scenario_", "", regex=False)
        schedule = schedule[
            schedule["scenario_id"].isin(self.scenario_ids)
            & schedule["simyear"].isin(self.years)
            & (schedule["doy"] >= 1)
            & (schedule["doy"] <= 365)
        ]

        unknown_events = sorted(set(schedule["management"].unique()) - set(self.management_events))
        if unknown_events:
            print(f"Warning: ignoring unknown management events: {unknown_events}")
        schedule = schedule[schedule["management"].isin(self.management_events)]

        for row in schedule.itertuples(index=False):
            key = (str(row.scenario_id), int(row.simyear))
            if key not in management_by_key:
                continue
            event_idx = event_to_idx[row.management]
            management_by_key[key][int(row.doy) - 1, event_idx] = 1.0

        return management_by_key

    def _build_static_lookup(self) -> Dict[str, np.ndarray]:
        static_by_point: Dict[str, np.ndarray] = {}
        for point_id in self.point_ids:
            if point_id not in self.static_raw.index:
                raise KeyError(f"Point {point_id} is missing from {self.initial_conditions_path}")
            row = self.static_raw.loc[[point_id], self.static_columns]
            static_by_point[point_id] = self.normalizers["static"].transform_frame(row)[0].astype(np.float32)
        return static_by_point

    def _read_harvest_file(self, scenario_id: str) -> pd.DataFrame:
        harvest = pd.read_csv(self._harvest_path(scenario_id))
        harvest = harvest.rename(columns={"id": "point_id", "simyear": "Year", "dayofyr": "doy"})
        harvest["scenario_id"] = str(scenario_id)
        harvest["point_id"] = harvest["point_id"].astype(str)
        harvest["Year"] = harvest["Year"].astype(int)
        return harvest[["scenario_id", "point_id", "Year", "doy", "cgrain"]]

    def _read_monthly_file(self, scenario_id: str) -> pd.DataFrame:
        monthly = pd.read_csv(self._monthly_path(scenario_id))
        monthly = monthly.rename(columns={"id": "point_id", "simyear": "Year"})
        monthly["scenario_id"] = str(scenario_id)
        monthly["point_id"] = monthly["point_id"].astype(str)
        monthly["Year"] = monthly["Year"].astype(int)
        return monthly[["scenario_id", "point_id", "Year", "month", "somsc"]]

    def _load_outputs(self) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float]]:
        output_by_key: Dict[Tuple[str, str, int], float] = {}
        mask_by_key: Dict[Tuple[str, str, int], float] = {}

        for scenario_id in self.scenario_ids:
            harvest = self._read_harvest_file(scenario_id)
            harvest = harvest[
                harvest["point_id"].isin(self.point_ids)
                & harvest["Year"].isin(self.years)
            ]
            harvest = harvest.sort_values(["scenario_id", "point_id", "Year", "doy"])
            harvest = harvest.drop_duplicates(["scenario_id", "point_id", "Year"], keep="first")

            for row in harvest.itertuples(index=False):
                key = (str(row.scenario_id), str(row.point_id), int(row.Year))
                if pd.isna(row.cgrain):
                    continue
                output_by_key[key] = self.normalizers["output"].transform_scalar(float(row.cgrain))
                mask_by_key[key] = 1.0

        return output_by_key, mask_by_key

    @staticmethod
    def _build_doy_features() -> np.ndarray:
        doy = np.arange(1, 366, dtype=np.float32).reshape(-1, 1)
        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)
        return np.concatenate([doy_sin, doy_cos], axis=1).astype(np.float32)

class DayCentSOMSCDataset(Dataset):
    """
    SOMSC-focused DayCent Dataset for one split.

    Samples are keyed by ``(scenario_id, year, month, point_id)`` and returned as a
    dictionary suitable for PyTorch workflows.
    """

    def __init__(self, params: Any, split: str = "train"):
        self.config = _params_to_dict(params)
        self.split = split
        self.data_root = self.config.get("data_root", DEFAULT_DATA_ROOT)
        self.data_cfg = self.config.get("data", {})
        self.splits_cfg = self.config.get("splits", {})
        self.normalization_cfg = self.config.get("normalization", {})

        if split not in self.splits_cfg:
            available = ", ".join(sorted(self.splits_cfg))
            raise KeyError(f"Unknown split {split!r}. Available splits: {available}")

        self.input_dir = os.path.join(self.data_root, "InputData")
        self.weather_dir = os.path.join(self.input_dir, "WeatherData")
        self.output_dir = os.path.join(self.data_root, "OutputData_Synthetic_10000")
        self.lookup_path = os.path.join(self.input_dir, "sample_points_lookup_SAS.xlsx")
        self.initial_conditions_path = os.path.join(self.input_dir, "initial_site_conditions.xlsx")
        self.schedule_path = os.path.join(self.input_dir, "schedule_scenarios_all_Synthetic_10000.csv")

        self.weather_columns = list(self.data_cfg.get("weather_columns", ["Tmax", "Tmin", "Precip"]))
        self.management_events = list(self.data_cfg.get("management_events", []))
        if not self.management_events:
            raise ValueError("config.data.management_events must define a stable event vocabulary")

        self.all_point_ids = self._load_all_point_ids()
        self.scenario_ids, self.point_ids, self.years, self.months = self._resolve_split(split)
        self.train_scenario_ids, self.train_point_ids, self.train_years, self.train_months = self._resolve_split(
            self.normalization_cfg.get("fit_split", "train")
        )

        self._validate_paths()
        self._validate_scenario_files(self.scenario_ids)
        self._validate_scenario_files(self.train_scenario_ids)

        self.static_raw, self.static_columns = self._load_initial_conditions()
        self.normalizers = self._fit_train_normalizers()

        self.feature_names = (
            self.weather_columns
            + [f"event_{event}" for event in self.management_events]
            + ["doy_sin", "doy_cos"]
        )
        self.static_feature_names = self.static_columns
        self.doy_features = self._build_doy_features()

        self.weather_by_key = self._load_weather()
        self.management_by_key = self._load_management()
        self.static_by_point = self._build_static_lookup()
        self.output_by_key, self.output_mask_by_key = self._load_outputs()

        self.sample_keys = list(itertools.product(self.scenario_ids, self.years, self.months, self.point_ids))

    def __len__(self) -> int:
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        scenario_id, year, month, point_id = self.sample_keys[idx]
        weather = self.weather_by_key[(point_id, year)]
        management = self.management_by_key[(scenario_id, year)]
        daily_input = np.concatenate([weather, management, self.doy_features], axis=1).astype(np.float32)

        output_key = (scenario_id, point_id, year, month)
        output = self.output_by_key.get(output_key, 0.0)
        output_mask = self.output_mask_by_key.get(output_key, 0.0)

        return {
            "input": torch.tensor(daily_input, dtype=torch.float32),
            "static": torch.tensor(self.static_by_point[point_id], dtype=torch.float32),
            "output": torch.tensor(output, dtype=torch.float32),
            "output_mask": torch.tensor(output_mask, dtype=torch.float32),
            "scenario_id": scenario_id,
            "point_id": point_id,
            "year": str(year),
            "month": str(month),
            "global_idx": idx,
        }

    def denormalize_weather(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized weather columns back to native data units."""
        return self.normalizers["weather"].inverse_array(values)

    def denormalize_output(self, value: float) -> float:
        """Convert normalized ``SOMSC`` back to mg C kg−1 monthly-1."""
        return self.normalizers["output"].inverse_scalar(value)

    def _load_all_point_ids(self) -> List[str]:
        lookup = pd.read_excel(self.lookup_path, sheet_name="sample_points_lookup_SAS")
        ids = lookup["id"].astype(str).tolist()
        if self.data_cfg.get("point_sort", "numeric") == "numeric":
            return sorted(ids, key=lambda x: int(x))
        return sorted(ids)

    def _resolve_split(self, split: str) -> Tuple[List[str], List[str], List[int]]:
        split_cfg = self.splits_cfg[split]
        scenarios = [str(s) for s in split_cfg.get("scenarios", [])]
        years = _range_or_list(split_cfg, "years", "year_range")
        months = _range_or_list(split_cfg, "months", "month_range")

        if "points" in split_cfg and split_cfg["points"] is not None:
            point_ids = [str(p) for p in split_cfg["points"]]
        elif "point_slice" in split_cfg and split_cfg["point_slice"] is not None:
            start = int(split_cfg["point_slice"]["start"])
            end = int(split_cfg["point_slice"]["end"])
            point_ids = self.all_point_ids[start:end]
        else:
            raise ValueError(f"Split {split!r} must define either points or point_slice")

        if not scenarios or not years or not point_ids or not months:
            raise ValueError(
                f"Split {split!r} resolved to scenarios={len(scenarios)}, "
                f"points={len(point_ids)}, years={len(years)}"
            )
        return scenarios, point_ids, years, months

    def _validate_paths(self) -> None:
        required_paths = [
            self.data_root,
            self.input_dir,
            self.weather_dir,
            self.output_dir,
            self.lookup_path,
            self.initial_conditions_path,
            self.schedule_path,
        ]
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError("Missing required DayCent paths:\n" + "\n".join(missing))

    def _validate_scenario_files(self, scenario_ids: List[str]) -> None:
        missing = []
        validate_monthly = bool(self.data_cfg.get("validate_monthly_files", True))
        for scenario_id in scenario_ids:
            harvest_path = self._harvest_path(scenario_id)
            monthly_path = self._monthly_path(scenario_id)
            if not os.path.exists(harvest_path):
                missing.append(harvest_path)
            if validate_monthly and not os.path.exists(monthly_path):
                missing.append(monthly_path)
        if missing:
            raise FileNotFoundError(
                "Missing DayCent scenario output files. "
                "The old synthetic loader expects SAS_scenario_{id}_harvest.csv "
                "and SAS_scenario_{id}_monthly.csv files:\n"
                + "\n".join(missing)
            )

    def _harvest_path(self, scenario_id: str) -> str:
        return os.path.join(self.output_dir, f"SAS_scenario_{scenario_id}_harvest.csv")

    def _monthly_path(self, scenario_id: str) -> str:
        return os.path.join(self.output_dir, f"SAS_scenario_{scenario_id}_monthly.csv")

    def _load_initial_conditions(self) -> Tuple[pd.DataFrame, List[str]]:
        frame = pd.read_excel(self.initial_conditions_path, sheet_name="initial_conditions")
        frame["id"] = frame["id"].astype(str)
        frame = frame.set_index("id")
        frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame, frame.columns.tolist()

    def _fit_train_normalizers(self) -> Dict[str, ZScoreNormalizer]:
        min_std = float(self.normalization_cfg.get("min_std", 1.0e-6))

        weather_frames = []
        for point_id in self.train_point_ids:
            weather_frames.append(
                self._read_weather_file(point_id, years=self.train_years, normalize=False)
            )
        weather_train = pd.concat(weather_frames, ignore_index=True)
        weather_norm = ZScoreNormalizer.fit(weather_train, self.weather_columns, min_std=min_std)

        static_train = self.static_raw.loc[self.train_point_ids, self.static_columns]
        static_norm = ZScoreNormalizer.fit(static_train, self.static_columns, min_std=min_std)

        output_values = []
        for scenario_id in self.train_scenario_ids:
            monthly = self._read_monthly_file(scenario_id)
            mask = (
                monthly["point_id"].isin(self.train_point_ids)
                & monthly["Year"].isin(self.train_years)
            )
            output_values.append(monthly.loc[mask, ["somsc"]])
        output_train = pd.concat(output_values, ignore_index=True)
        if output_train["somsc"].dropna().empty:
            raise ValueError("No training somsc values found for output normalization")
        output_norm = ZScoreNormalizer.fit(output_train, ["somsc"], min_std=min_std)

        return {
            "weather": weather_norm,
            "static": static_norm,
            "output": output_norm,
        }

    def _read_weather_file(self, point_id: str, years: List[int], normalize: bool) -> pd.DataFrame | Dict[int, np.ndarray]:
        path = os.path.join(self.weather_dir, f"{point_id}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weather file for point {point_id}: {path}")

        usecols = ["Year", "doy"] + self.weather_columns
        frame = pd.read_csv(path, usecols=usecols)
        frame = frame[frame["Year"].isin(years)]
        frame = frame[(frame["doy"] >= 1) & (frame["doy"] <= 365)]
        frame = frame.sort_values(["Year", "doy"])

        if not normalize:
            return frame

        by_year: Dict[int, np.ndarray] = {}
        for year in years:
            year_frame = frame[frame["Year"] == year].drop_duplicates("doy", keep="last")
            year_frame = year_frame.set_index("doy").reindex(range(1, 366))
            values = year_frame[self.weather_columns].to_numpy(dtype=np.float32)
            by_year[year] = self.normalizers["weather"].transform_array(values).astype(np.float32)
        return by_year

    def _load_weather(self) -> Dict[Tuple[str, int], np.ndarray]:
        weather_by_key: Dict[Tuple[str, int], np.ndarray] = {}
        for point_id in self.point_ids:
            point_weather = self._read_weather_file(point_id, self.years, normalize=True)
            for year, values in point_weather.items():
                weather_by_key[(point_id, year)] = values
        return weather_by_key

    def _load_management(self) -> Dict[Tuple[str, int], np.ndarray]:
        management_by_key = {
            (scenario_id, year): np.zeros((365, len(self.management_events)), dtype=np.float32)
            for scenario_id in self.scenario_ids
            for year in self.years
        }
        event_to_idx = {event: idx for idx, event in enumerate(self.management_events)}

        schedule = pd.read_csv(
            self.schedule_path,
            usecols=["scenario", "simyear", "doy", "management"],
        )
        schedule["scenario_id"] = schedule["scenario"].astype(str).str.replace("scenario_", "", regex=False)
        schedule = schedule[
            schedule["scenario_id"].isin(self.scenario_ids)
            & schedule["simyear"].isin(self.years)
            & (schedule["doy"] >= 1)
            & (schedule["doy"] <= 365)
        ]

        unknown_events = sorted(set(schedule["management"].unique()) - set(self.management_events))
        if unknown_events:
            print(f"Warning: ignoring unknown management events: {unknown_events}")
        schedule = schedule[schedule["management"].isin(self.management_events)]

        for row in schedule.itertuples(index=False):
            key = (str(row.scenario_id), int(row.simyear))
            if key not in management_by_key:
                continue
            event_idx = event_to_idx[row.management]
            management_by_key[key][int(row.doy) - 1, event_idx] = 1.0

        return management_by_key

    def _build_static_lookup(self) -> Dict[str, np.ndarray]:
        static_by_point: Dict[str, np.ndarray] = {}
        for point_id in self.point_ids:
            if point_id not in self.static_raw.index:
                raise KeyError(f"Point {point_id} is missing from {self.initial_conditions_path}")
            row = self.static_raw.loc[[point_id], self.static_columns]
            static_by_point[point_id] = self.normalizers["static"].transform_frame(row)[0].astype(np.float32)
        return static_by_point

    def _read_harvest_file(self, scenario_id: str) -> pd.DataFrame:
        harvest = pd.read_csv(self._harvest_path(scenario_id))
        harvest = harvest.rename(columns={"id": "point_id", "simyear": "Year", "dayofyr": "doy"})
        harvest["scenario_id"] = str(scenario_id)
        harvest["point_id"] = harvest["point_id"].astype(str)
        harvest["Year"] = harvest["Year"].astype(int)
        return harvest[["scenario_id", "point_id", "Year", "doy", "cgrain"]]

    def _read_monthly_file(self, scenario_id: str) -> pd.DataFrame:
        monthly = pd.read_csv(self._monthly_path(scenario_id))
        monthly = monthly.rename(columns={"id": "point_id", "simyear": "Year"})
        monthly["scenario_id"] = str(scenario_id)
        monthly["point_id"] = monthly["point_id"].astype(str)
        monthly["Year"] = monthly["Year"].astype(int)
        return monthly[["scenario_id", "point_id", "Year", "month", "somsc"]]

    def _load_outputs(self) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float]]:
        output_by_key: Dict[Tuple[str, str, int], float] = {}
        mask_by_key: Dict[Tuple[str, str, int], float] = {}

        for scenario_id in self.scenario_ids:
            monthly = self._read_monthly_file(scenario_id)
            monthly = monthly[
                monthly["point_id"].isin(self.point_ids)
                & monthly["Year"].isin(self.years)
            ]
            monthly = monthly.sort_values(["scenario_id", "point_id", "Year", "month"])
            monthly = monthly.drop_duplicates(["scenario_id", "point_id", "Year", "month"], keep="first")

            for row in monthly.itertuples(index=False):
                key = (str(row.scenario_id), str(row.point_id), int(row.Year), int(row.month))
                if pd.isna(row.somsc):
                    continue
                output_by_key[key] = self.normalizers["output"].transform_scalar(float(row.somsc))
                mask_by_key[key] = 1.0

        return output_by_key, mask_by_key

    @staticmethod
    def _build_doy_features() -> np.ndarray:
        doy = np.arange(1, 366, dtype=np.float32).reshape(-1, 1)
        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)
        return np.concatenate([doy_sin, doy_cos], axis=1).astype(np.float32)


def get_data_loader(params: Any, split: str = "train", shuffle: bool | None = None) -> Tuple[DataLoader, DayCentSOMSCDataset]:
    """
    Create a PyTorch DataLoader and the underlying DayCentDataset.
    """
    cfg = _params_to_dict(params)
    if params.get("task") == "SOMSC":
        dataset = DayCentSOMSCDataset(params, split=split)
    else:
        dataset = DayCentYieldDataset(params, split=split)

    if shuffle is None:
        shuffle = bool(cfg.get("shuffle", split == "train"))

    num_workers = int(cfg.get("num_data_workers", 0))
    prefetch_factor = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("local_batch_size", 32)),
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init if num_workers > 0 else None,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def print_dataset_summary(config, dataset, loader) -> None:
    sample = dataset[0]
    print("\n========== DAYCENT DATASET INFORMATION ==========")
    print(f"Split:              {dataset.split}")
    print(f"Samples:            {len(dataset)}")
    print(f"Scenarios:          {dataset.scenario_ids}")
    print(f"Points:             {len(dataset.point_ids)}")
    print(f"Years:              {dataset.years[0]}-{dataset.years[-1]} ({len(dataset.years)})")
    print(f"Batch size:         {loader.batch_size}")
    print(f"Number of batches:  {len(loader)}")
    print("\nShapes:")
    print(f"  input:   {tuple(sample['input'].shape)}  [days, features]")
    print(f"  static:  {tuple(sample['static'].shape)}")
    print(f"  output:  {tuple(sample['output'].shape)}")
    print("\nFeature counts:")
    print(f"  weather:      {len(dataset.weather_columns)}")
    print(f"  management:   {len(dataset.management_events)}")
    print("  doy encoding: 2")
    print(f"  total input:  {len(dataset.feature_names)}")
    print(f"  static:       {len(dataset.static_feature_names)}")
    print("\nFirst sample:")
    if config.get("task") == "SOMSC":
        raw_somsc = dataset.denormalize_output(sample["output"].item())
        print(
            f"  scenario={sample['scenario_id']} point={sample['point_id']} "
            f"year={sample['year']} month={sample['month']} SOMSC={raw_somsc:.3f} "
            f"mask={sample['output_mask'].item():.0f}"
        )
    else:
        raw_yield = dataset.denormalize_output(sample["output"].item())
        print(
            f"  scenario={sample['scenario_id']} point={sample['point_id']} "
            f"year={sample['year']} cgrain={raw_yield:.3f} "
            f"mask={sample['output_mask'].item():.0f}"
        )
    print("=================================================\n")


if __name__ == "__main__":
    from utils.YParams import YParams

    parser = argparse.ArgumentParser(description="Smoke test the old-format DayCent data loader")
    parser.add_argument("--yaml_config", default="config.yaml", type=str)
    parser.add_argument("--config", default="demo_yield", type=str)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"], type=str)
    parser.add_argument("--num_batches", default=2, type=int)
    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config)

    start_time = time.time()
    dataloader, dataset = get_data_loader(params, split=args.split, shuffle=False)
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds")

    print_dataset_summary(params, dataset, dataloader)

    print("Testing batch loading...")
    for batch_idx, batch in enumerate(dataloader):
        print(
            f"Batch {batch_idx}: input={tuple(batch['input'].shape)} "
            f"static={tuple(batch['static'].shape)} output={tuple(batch['output'].shape)} "
            f"first=({batch['scenario_id'][0]}, {batch['point_id'][0]}, {batch['year'][0]})"
        )
        if batch_idx + 1 >= args.num_batches:
            break

    print("Data loader smoke test completed successfully.")
