# DADT

This repository contains a PyTorch implementation of the UDAR-based time-series anomaly detection pipeline used in this project.

## Overview

The codebase provides:

- A single entrypoint for training and testing: `main.py`
- A solver for training, validation, and evaluation: `solver.py`
- The UDAR model implementation: `model/udar.py`
- Dataset loaders for NPY-based and CSV-based anomaly detection benchmarks
- Shell scripts for running training and direct evaluation on supported datasets

## Project Structure

```text
DADT/
├── main.py
├── solver.py
├── model/
│   ├── attn.py
│   ├── embed.py
│   └── udar.py
├── data_factory/
│   └── data_loader.py
├── metrics/
├── dataset/
├── scripts/
│   ├── start.sh
│   ├── test.sh
│   ├── MSL.sh
│   ├── NIPS_TS_Swan.sh
│   ├── NIPS_TS_Water.sh
│   ├── PSM.sh
│   ├── SWAT.sh
│   └── WADI.sh
└── checkpoints/
```

## Requirements

Recommended environment:

- Python 3.8+
- PyTorch
- NumPy
- pandas
- scikit-learn

Install the required packages with your preferred environment manager.

## Data Layout

Place datasets under `dataset/`.

Supported NPY-style datasets:

- `MSL`
- `SMAP`
- `NIPS_TS_Water`
- `NIPS_TS_Swan`
- `PSM`
- `SWAT`
- `WADI`

Expected NPY file naming:

```text
dataset/<DATASET_NAME>/
├── <DATASET_NAME>_train.npy
├── <DATASET_NAME>_test.npy
└── <DATASET_NAME>_test_label.npy
```

For CSV-style datasets, the loader expects:

```text
dataset/<DATASET_NAME>/
├── train.csv
├── test.csv
└── test_label.csv
```

## Training

Run a single experiment with:

```bash
python main.py \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --anormly_ratio 1.0 \
  --batch_size 256 \
  --win_size 110 \
  --decomp_kernel 25 \
  --period 15 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train
```

`train` mode runs training first and then evaluates the saved checkpoint.

You can also use the provided script:

```bash
bash scripts/start.sh
```

## Testing

Run direct evaluation with:

```bash
python main.py \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --anormly_ratio 1.0 \
  --batch_size 256 \
  --win_size 110 \
  --decomp_kernel 25 \
  --period 15 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode test
```

To evaluate all configured datasets:

```bash
bash scripts/test.sh
```

## Checkpoints and Outputs

Model checkpoints and test reports are written under `checkpoints/` using per-run subdirectories:

```text
checkpoints/<DATASET_NAME>_run_id=<RUN_ID>/
```

Each run may include:

- A saved checkpoint: `<DATASET_NAME>_run<RUN_ID>_checkpoint.pth`
- A text evaluation report: `<DATASET_NAME>_run<RUN_ID>_test_results.txt`

## Notes

- The current implementation uses the full UDAR pipeline without ablation switches.
- `run_id` is kept for checkpoint naming and repeatable experiment organization.
- Dataset-specific training scripts are available in `scripts/` for convenience.
