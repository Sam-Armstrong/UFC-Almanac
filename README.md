# UFC-Almanac

<div align="center">
  <img src="images/ufc_almanac.png" alt="UFC Almanac" width="400" />
</div>

A collection of datasets of UFC fight results, fight stats (both updated weekly) and fighter data (updated monthly),
containing the data for all UFC fights since 2010.
Also contains code for transforming the data into formats for training machine learning models,
and training scripts for a variety of deep learning models.

## Next UFC Event Predictions

Event date: July 11, 2026
| Fight | Win | Loss | Draw |
| --- | --- | --- | --- |
| Conor McGregor vs Max Holloway | 27.1% | 72.7% | 0.3% |
| Benoit Saint Denis vs Paddy Pimblett | 54.4% | 44.8% | 0.8% |
| Cory Sandhagen vs Mario Bautista | 42.1% | 57.3% | 0.6% |
| Brandon Royval vs Lone'er Kavanagh | 53.6% | 46.0% | 0.4% |
| King Green vs Terrance McKinney | 72.1% | 27.6% | 0.4% |
| Nikita Krylov vs Robert Whittaker | 55.5% | 44.1% | 0.5% |
| Cody Garbrandt vs Adrian Yanez | 52.2% | 47.6% | 0.3% |
| Tracy Cortez vs Wang Cong | 53.1% | 46.1% | 0.8% |
| Alessandro Costa vs Cody Durden | 67.2% | 32.5% | 0.3% |


The model used for these predictions is a models/transformer_model.py trained using the following command:
```bash
ufc-train --model transformer --path artifacts/core/transformer_model.pt --epochs 30 --dropout 0.5 --weight-decay 3e-5 --learning-rate 3e-5 --d-model 128 --val-fraction 0.1
```

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

#### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model architecture (`linear`, `mlp`, or `transformer`) | `linear` |
| `--epochs` | Number of training epochs | `50` |
| `--batch-size` | Training batch size | `256` |
| `--learning-rate` | Adam learning rate | `3e-4` |
| `--val-fraction` | Fraction of data held out for validation | `0.1` |
| `--weight-decay` | L2 regularization strength for Adam | `1e-4` |
| `--dropout` | Dropout probability | `0.2` |
| `--d-model` | Transformer hidden dimension (`transformer` only) | `64` |
| `--num-layers` | Number of transformer encoder layers (`transformer` only) | `2` |
| `--max-fights` | Past fights per fighter / sequence length (`transformer` only) | `8` |
| `--path` | Path to save trained model weights (normalization stats saved alongside as `<stem>_normalization.pt`) | `artifacts/checkpoints/<ModelName>.pt` |
| `--rebuild-data` | Regenerate training tensors from CSV files before training | off |

Use `--rebuild-data` when the underlying CSV data has been updated. Changing `--max-fights` also regenerates transformer training data when it does not match the saved tensors.

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
| `--path` | Path to trained model weights | `artifacts/checkpoints/<ModelName>.pt` |

The predictor loads trained weights and normalization stats from `artifacts/checkpoints/` for the selected model by default. Train a model first with `ufc-train`, or pass `--path` to load a custom checkpoint.

## Automated data updates

GitHub Actions workflows scrape new fight data weekly and fighter data monthly,
adding new data to the csv files in the `data/` directory.
