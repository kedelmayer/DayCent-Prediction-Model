from __future__ import annotations
import argparse
import os
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from utils.YParams import YParams

from data_loader import DayCentYieldDataset, DayCentSOMSCDataset


# Generate README-ready overview plots for the DayCent starter repo.
# Run from this directory: python dataset_plots.py --config demo_yield


FIGURE_DIR = "docs/figures"


def _load_config(yaml_path: str, config_name: str) -> Dict:
    with open(yaml_path) as f:
        configs = yaml.safe_load(f)
    if config_name not in configs:
        raise KeyError(f"Config {config_name!r} not found in {yaml_path}")
    return configs[config_name]


def _sorted_point_ids(lookup: pd.DataFrame) -> List[str]:
    return sorted(lookup["id"].astype(str).tolist(), key=lambda x: int(x))


def _split_points(split_cfg: Dict, all_point_ids: List[str]) -> List[str]:
    if split_cfg.get("points") is not None:
        return [str(point_id) for point_id in split_cfg["points"]]
    point_slice = split_cfg["point_slice"]
    return all_point_ids[int(point_slice["start"]) : int(point_slice["end"])]


def plot_demo_split_map(config: Dict, save_path: str) -> None:
    lookup_path = os.path.join(
        config["data_root"],
        "InputData",
        "sample_points_lookup_SAS.xlsx",
    )
    lookup = pd.read_excel(lookup_path, sheet_name="sample_points_lookup_SAS")
    lookup["id"] = lookup["id"].astype(str)
    all_point_ids = _sorted_point_ids(lookup)

    split_points = {
        split: set(_split_points(split_cfg, all_point_ids))
        for split, split_cfg in config["splits"].items()
    }

    colors = {
        "train": "#2563eb",
        "valid": "#f59e0b",
        "test": "#dc2626",
    }

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(
        lookup["POINT_X"],
        lookup["POINT_Y"],
        s=16,
        color="#cbd5e1",
        label="unused points",
        alpha=0.75,
    )

    for split in ["train", "valid", "test"]:
        split_lookup = lookup[lookup["id"].isin(split_points[split])]
        ax.scatter(
            split_lookup["POINT_X"],
            split_lookup["POINT_Y"],
            s=54,
            color=colors[split],
            edgecolor="white",
            linewidth=0.8,
            label=f"{split} ({len(split_lookup)} points)",
            zorder=3,
        )

    ax.set_title("Demo split point locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved split map to: {save_path}")


def _collect_train_arrays(dataset, task_somsc: bool) -> Dict[str, np.ndarray]:
    weather_values = []
    output_rows = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        daily = sample["input"].numpy()
        weather = dataset.denormalize_weather(daily[:, : len(dataset.weather_columns)])
        weather_values.append(weather)

        if sample["output_mask"].item() > 0:
            if task_somsc:
                output_rows.append(
                {
                    "scenario_id": sample["scenario_id"],
                    "year": int(sample["year"]),
                    "month": int(sample["month"]),
                    "somsc": dataset.denormalize_output(sample["output"].item()),
                })
            else:
                output_rows.append(
                {
                    "scenario_id": sample["scenario_id"],
                    "year": int(sample["year"]),
                    "cgrain": dataset.denormalize_output(sample["output"].item()),
                })

    return {
        "weather": np.stack(weather_values, axis=0),
        "outputs": pd.DataFrame(output_rows),
    }


def plot_weather_output_overview(dataset, save_path: str, task_somsc: bool) -> None:
    arrays = _collect_train_arrays(dataset, task_somsc)
    weather = arrays["weather"]
    outputs = arrays["outputs"]
    days = np.arange(1, 366)

    weather_mean = weather.mean(axis=0)
    weather_q25 = np.quantile(weather, 0.25, axis=0)
    weather_q75 = np.quantile(weather, 0.75, axis=0)

    scenarios = sorted(outputs["scenario_id"].unique(), key=lambda x: int(x))

    if task_somsc:
        somsc_by_scenario = [
            outputs.loc[outputs["scenario_id"] == scenario, "somsc"].to_numpy()
            for scenario in scenarios
        ]
    else:
        yield_by_scenario = [
            outputs.loc[outputs["scenario_id"] == scenario, "cgrain"].to_numpy()
            for scenario in scenarios
        ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    axes[0].plot(days, weather_mean[:, 0], label="Tmax mean", color="#c2410c", linewidth=1.4)
    axes[0].fill_between(days, weather_q25[:, 0], weather_q75[:, 0], color="#fed7aa", alpha=0.45)
    axes[0].plot(days, weather_mean[:, 1], label="Tmin mean", color="#1d4ed8", linewidth=1.4)
    axes[0].fill_between(days, weather_q25[:, 1], weather_q75[:, 1], color="#bfdbfe", alpha=0.45)
    precip_ax = axes[0].twinx()
    precip_ax.plot(days, weather_mean[:, 2], label="Precip mean", color="#0f766e", linewidth=1.0, alpha=0.75)
    axes[0].set_title("Train weather seasonality")
    axes[0].set_xlabel("Day of year")
    axes[0].set_ylabel("Temperature (deg C)")
    precip_ax.set_ylabel("Precip (source units)")
    axes[0].grid(alpha=0.25)
    lines, labels = axes[0].get_legend_handles_labels()
    precip_lines, precip_labels = precip_ax.get_legend_handles_labels()
    axes[0].legend(lines + precip_lines, labels + precip_labels, frameon=False, loc="upper left")

    if task_somsc:
        box = axes[1].boxplot(
            somsc_by_scenario,
            tick_labels=[f"{s}" for s in scenarios],
            patch_artist=True,
            showfliers=False,
        )
    else:
        box = axes[1].boxplot(
            yield_by_scenario,
            tick_labels=[f"{s}" for s in scenarios],
            patch_artist=True,
            showfliers=False,
        )
    for patch, color in zip(box["boxes"], ["#93c5fd", "#86efac", "#fca5a5"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    if task_somsc:
        axes[0].set_ylabel("Train soil organic carbon by scenario")
        axes[1].set_ylabel("SOMSC (mg C kg−1 monthly-1)")
    else:
        axes[0].set_ylabel("Train yield by scenario")
        axes[1].set_ylabel("cgrain (g C m-2 harvest-1)")
    axes[1].set_xlabel("Synthetic management scenario")
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved weather/output overview to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate README overview figures")
    parser.add_argument("--yaml_config", default="config.yaml", type=str)
    parser.add_argument("--config", default="demo_yield", type=str)
    args = parser.parse_args()

    os.makedirs(FIGURE_DIR, exist_ok=True)

    params = YParams(args.yaml_config, args.config)
    config = _load_config(args.yaml_config, args.config)

    if params.get("task") == "SOMSC":
        train_dataset = DayCentSOMSCDataset(config, split="train")
        task_somsc = True
    else:
        train_dataset = DayCentYieldDataset(config, split="train")
        task_somsc = False

    plot_demo_split_map(config, os.path.join(FIGURE_DIR, "daycent_demo_split_map.png"))
    plot_weather_output_overview(
        train_dataset,
        os.path.join(FIGURE_DIR, "daycent_train_weather_output_overview.png"),
        task_somsc
    )

    print(f"README plots are ready in: {FIGURE_DIR}")
