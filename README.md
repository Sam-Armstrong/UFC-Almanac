# UFC-Fight-Predictor

A store of datasets of UFC fight results, fight stats (both updated weekly) and fighter data (updated monthly),
containing the data for all UFC fights since 2010.
Also contains code for transforming the data into formats for training machine learning models,
and training scripts for a variety of deep learning models.

## Setup

```bash
pip install -e ".[all]"
```

To scrape data, install the Playwright browser:

```bash
playwright install chromium
```

## Usage

After installing the project, you can run commands either via the console scripts or the `scripts/` entrypoints.

### Scrape data (into data/*.csv files)

```bash
# Scrape fight results and stats
ufc-scrape-fights
# or
python scripts/scrape_fights.py

# Scrape fighter profiles
ufc-scrape-fighters
# or
python scripts/scrape_fighters.py
```

### Training

```bash
ufc-train --model transformer --rebuild-data
# or
python scripts/train.py --model transformer
```

#### Tunable parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model architecture (`linear`, `mlp`, or `transformer`) | `linear` |
| `--epochs` | Number of training epochs | `50` |
| `--batch-size` | Training batch size | `256` |
| `--learning-rate` | Adam learning rate | `3e-4` |
| `--val-fraction` | Fraction of data held out for validation | `0.1` |
| `--weight-decay` | L2 regularization strength for Adam | `1e-4` |
| `--dropout` | Dropout probability | `0.2` |
| `--rebuild-data` | Regenerate training tensors from CSV files before training | off |

Use `--rebuild-data` when the underlying CSV data has been updated.

### Inference (predict fight outcomes)

Interactive CLI:

```bash
ufc-predict --model linear
# or
python scripts/predict.py --model mlp
```

Enter two fighter names when prompted. Type `exit`, `quit`, or `q` to stop.

#### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model architecture to load: `linear`, `mlp`, or `transformer` | `linear` |

The predictor loads trained weights from `artifacts/checkpoints/` for the selected model. Train a model first with `ufc-train`.

## Automated data updates

GitHub Actions workflows scrape new fight data weekly and fighter data monthly,
adding new data to the csv files in the `data/` directory.
