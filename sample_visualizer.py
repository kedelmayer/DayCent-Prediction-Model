from __future__ import annotations
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_data_loader, print_dataset_summary
from utils.YParams import YParams


# Example visualizations for a single sample.
# Run from this directory: python sample_visualizer.py --config demo_yield --split train --sample_idx 0


OUTPUT_DIR = "visualization_outputs"


def _event_display_name(name: str) -> str:
    return name.replace("nitrogen_fertilization_", "N_").replace("_", " ")


def plot_sample_overview(dataset, sample_idx: int, save_path: str) -> None:
    sample = dataset[sample_idx]
    daily = sample["input"].numpy()
    weather = dataset.denormalize_weather(daily[:, : len(dataset.weather_columns)])
    event_start = len(dataset.weather_columns)
    event_end = event_start + len(dataset.management_events)
    events = daily[:, event_start:event_end]
    days = np.arange(1, 366)

    raw_yield = dataset.denormalize_output(sample["output"].item())

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [1.2, 0.8, 1.4]},
        constrained_layout=True,
    )

    axes[0].plot(days, weather[:, 0], label="Tmax", color="#c2410c", linewidth=1.4)
    axes[0].plot(days, weather[:, 1], label="Tmin", color="#1d4ed8", linewidth=1.4)
    axes[0].set_ylabel("Temperature")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].bar(days, weather[:, 2], width=1.0, color="#0f766e")
    axes[1].set_ylabel("Precip")
    axes[1].grid(alpha=0.25)

    axes[2].imshow(events.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    axes[2].set_yticks(np.arange(len(dataset.management_events)))
    axes[2].set_yticklabels([_event_display_name(e) for e in dataset.management_events], fontsize=8)
    axes[2].set_xlabel("Day of year")
    axes[2].set_ylabel("Management event")

    fig.suptitle(
        "DayCent sample: "
        f"scenario {sample['scenario_id']} | point {sample['point_id']} | "
        f"year {sample['year']} | cgrain {raw_yield:.2f} g C m-2",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved sample overview to: {save_path}")


def plot_yield_distribution(dataset, save_path: str) -> None:
    values = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["output_mask"].item() > 0:
            values.append(dataset.denormalize_output(sample["output"].item()))

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(values, bins=30, color="#4f46e5", alpha=0.85)
    ax.set_xlabel("cgrain (g C m-2 harvest-1)")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Yield distribution for {dataset.split} split")
    ax.grid(alpha=0.25)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved yield distribution to: {save_path}")

def plot_somsc_distribution(dataset, save_path: str) -> None:
    values = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["output_mask"].item() > 0:
            values.append(dataset.denormalize_output(sample["output"].item()))

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(values, bins=30, color="#4f46e5", alpha=0.85)
    ax.set_xlabel("SOMSC (mg C kg−1 monthly-1)")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Soil Organic Material/Soil Carbon distribution for {dataset.split} split")
    ax.grid(alpha=0.25)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved soil carbon distribution to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DayCent loader samples")
    parser.add_argument("--yaml_config", default="config.yaml", type=str)
    parser.add_argument("--config", default="demo_yield", type=str)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"], type=str)
    parser.add_argument("--sample_idx", default=0, type=int)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params = YParams(args.yaml_config, args.config)
    params.local_batch_size = 1
    params.num_data_workers = 0

    dataloader, dataset = get_data_loader(params, split=args.split, shuffle=False)
    print_dataset_summary(params, dataset, dataloader)

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx must be in [0, {len(dataset) - 1}]")

    plot_sample_overview(
        dataset,
        sample_idx=args.sample_idx,
        save_path=os.path.join(OUTPUT_DIR, f"daycent_sample_{args.split}_{args.sample_idx}.png"),
    )
    if params.get("task") == "SOMSC":
        plot_somsc_distribution(
            dataset,
            save_path=os.path.join(OUTPUT_DIR, f"daycent_somsc_distribution_{args.split}.png"),
        )
    else:
        plot_yield_distribution(
            dataset,
            save_path=os.path.join(OUTPUT_DIR, f"daycent_yield_distribution_{args.split}.png"),
        )

    print(f"Visualization complete. Outputs are in: {OUTPUT_DIR}")
