# DayCent Agricultural Prediction Model 

An agricultural time series forecasting model implementing the LSTM architecture, used to predict annual grain yield and monthly soil carbon from DayCent site simulation data. Created for the University of Minnesota's AI for Earth Summer Program 2026, using the DayCent Data Loader found here: https://github.com/dwij2212/ai_for_earth_daycent

## Features

- **DayCent support**: Uses `InputData/` and `OutputData_Synthetic_10000/`.
- **Yield and soil carbon focused tasks**: Predict annual `cgrain` or monthly `SOMSC` from weather, management, and static soil/site state.
- **PyTorch integration**: Native `Dataset` and `DataLoader` support.
- **Leakage-safe normalization**: Weather, static variables, and yield are normalized from the train split only.
- **Deterministic demo subset**: Small fixed scenarios, points, and years for quick first runs.
- **Split and Sample visualizations**: Plot daily weather, management events, and yield/SOMSC distribution.
- **Validation and Test Prediction visualizations**: Plot training/validation for each epoch and test evaluation (model predictions vs. actual data) to show model performance.

## Repository Structure

```text
ai_for_earth_daycent/
├── data_loader.py                      # Core DayCent loading code
├── LSTM.py                             # Core DayCent prediction model
├── using_data_loader.py                # Dataset inspection and visualization examples
├── make_readme_plots.py                # Recreate README overview figures
├── docs/figures/                       # README-ready generated figures
├── config.yaml                         # Small deterministic demo config
├── requirements.txt
├── utils/
│   └── YParams.py
└── README.md
```

## Quick Start

Download the DayCent data tar file from Google Drive:

[Download the DayCent data tar archive](https://drive.google.com/file/d/1LdoYOwgbnD85t9CNlrExklF8p9mXLNaa/view?usp=drive_link)

Extract it somewhere outside this git repo:

```bash
mkdir -p /path/to/daycent/data
tar -xf /path/to/downloaded/<downloaded-file>.tar -C /path/to/daycent/data
```

Then update `data_root` in `config.yaml` to the extracted `SAS_KGML_090925` directory, for example:

```yaml
data_root: /path/to/daycent/data/SAS_KGML_090925
```

Set up a Python environment and install dependencies from `requirements.txt`:

```bash
cd ~/DayCent_Prediction_Model                  # Navigate to DayCent_Prediction_Model
python3 -m venv .venv
.venv\Scripts\activate.ps1
# For Linux, use:  source .venv/bin/activate
pip3 install -r requirements.txt
```

Run the loader smoke test:

```bash
py data_loader.py --config demo_yield --split train
```

Create a sample visualization:

```bash
py sample_visualizer.py --config demo_yield --split train --sample_idx 0
```

Generated plots are written to `visualization_outputs/` and ignored by git.

Create plots for dataset split information:

```bash
py dataset_plots.py --config demo_yield
```
Generated plots are written to `docs/figures` and ignored by git.

Run the model to predict annual yield using the model config:
```bash
py LSTM.py --config model_yield
```

Run the model to predict monthly soil organic carbon using the model config:
```bash
py LSTM.py --config model_SOMSC
```

## Dataset Details

This dataset documentation comes from the DayCent Data Loader repository found here: https://github.com/dwij2212/ai_for_earth_daycent
- **Overview**: Simulation data from the DayCent ecosystem model, used to simulate daily carbon, nitrogen, water, and plant-growth processes.
- **Sites**: 203 geographic points.
- **Time period**: weather from 2000-2024; the demo ML split uses 2002-2023.
- **Drivers**: daily weather, daily management events, and static initial soil/site conditions.
- **Target**: annual harvest `cgrain`, in `g C m-2 harvest-1`, monthly soil carbon `SOMSC`, in `mg C kg−1 monthly-1`

### Input Features

#### Daily Weather

- `Tmax`: daily maximum temperature.
- `Tmin`: daily minimum temperature.
- `Precip`: daily precipitation in the source DayCent units.

Weather controls crop stress, growing-season length, and water availability.

#### Daily Management Events

The starter loader encodes 14 event types as daily binary channels:

- Planting: corn, soybean, winter wheat, ryegrass.
- Tillage: conventional, reduced, no-till.
- Fertilizer: four nitrogen rates.
- Other markers: herbicide, harvest, cycle end.

Management defines what crop is grown, when it is planted and harvested, and how much nitrogen is available.

#### Static Site Conditions

Each point has 300 initial-condition variables describing soil carbon pools, nitrogen state, hydrology, and site properties. The loader standardizes these variables and returns them as a static vector.


## Technical Implementation

### Data Pipeline

```python
sample = dataset[0]
sample["input"].shape   # [365, 19] daily sequence
sample["static"].shape  # [300] initial conditions
sample["output"].shape  # [] normalized annual cgrain or monthly soil carbon
```

### Sample Format

Each dataset item is a dictionary:

```python
{
    "input": FloatTensor[365, F],
    "static": FloatTensor[300],
    "output": FloatTensor[],
    "output_mask": FloatTensor[],
    "scenario_id": str,
    "point_id": str,
    "year": str,
    "month": str,                   # For SOMSC-focused tasks only
    "global_idx": int,
}
```

The default daily feature size is `F = 19`:

- 3 normalized weather columns: `Tmax`, `Tmin`, `Precip`
- 14 binary management-event channels
- 2 seasonal encodings: `doy_sin`, `doy_cos`

For yield-focused tasks, `output` is normalized annual `cgrain`; use `dataset.denormalize_output(...)` to convert it back to `g C m-2 harvest-1`.
For SOMSC-focused tasks, `output` is normalized monthly `SOMSC`; use `dataset.denormalize_output(...)` to convert it back to `mg C kg−1 monthly-1`.

## Using Config.YAML
Data configurations can be modified in `config.yaml` to include more points, scenarios, or years, or to focus on different tasks.
- **data_root**: Must be updated with the path to your daycent data.
- **task**: Must yield or SOMSC.
- **scenarios**: May include any numbered scenario in the dataset.
- **month_range**: {start: 1, end: 12}, must be included for each split in a SOMSC-focused task.

### Demo Splits
The loader default `demo_yield` config uses complete synthetic scenarios:

| Split | Scenarios | Points | Years | Samples |
|---|---:|---:|---:|---:|
| Train | `3, 4, 12` | first 24 sorted point IDs | `2002-2020` | `1368` |
| Valid | `14` | next 8 sorted point IDs | `2021-2022` | `16` |
| Test | `18` | next 8 sorted point IDs | `2023` | `8` |

The model default `model_yield` config uses complete synthetic scenarios:

| Split | Scenarios | Points | Years | Samples |
|---|---:|---:|---:|---:|
| Train | `3, 4, 10, 12, 14, 16, 18, 20, 22, 26, 30, 35, 40, 42, 51` | first 60 sorted point IDs | `2002-2020` | `17100` |
| Valid | `100, 107, 119` | next 15 sorted point IDs | `2021-2022` | `150` |
| Test | `123, 134, 143` | next 15 sorted point IDs | `2023` | `45` |

The `model_SOMSC` config uses complete synthetic scenarios:

| Split | Scenarios | Points | Years | Samples |
|---|---:|---:|---:|---:|
| Train | `3, 4, 10, 12, 14, 16, 18, 20, 22, 26, 30, 35, 40, 42, 51` | first 10 sorted point IDs | `2002-2020` | `34200` |
| Valid | `100, 107, 119` | next 5 sorted point IDs | `2021-2022` | `360` |
| Test | `123, 134, 143` | next 5 sorted point IDs | `2023` | `180` |
