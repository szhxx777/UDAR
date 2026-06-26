#!/bin/bash

# Set the visible GPU device. Update this value if you use another GPU.
export CUDA_VISIBLE_DEVICES=2

echo "======================================================================="
echo "Starting execution of BEST hyperparameter combinations for all datasets"
echo "======================================================================="

# 1. MSL
# Best from search: win_size=110, decomp_kernel=25, period=15, k=3, lr=0.0001, epochs=3
echo ""
echo ">>> Running MSL (Input: 55, Output: 55, Ratio: 1.0)..."
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

# 2. NIPS_TS_Swan (Ref: Swan)
# Best from search: win_size=32, decomp_kernel=15, period=25, k=3, lr=0.0001, epochs=3
echo ""
echo ">>> Running NIPS_TS_Swan (Input: 38, Output: 38, Ratio: 0.5)..."
python main.py \
  --dataset NIPS_TS_Swan \
  --data_path dataset/NIPS_TS_Swan \
  --input_c 38 \
  --output_c 38 \
  --anormly_ratio 0.5 \
  --batch_size 256 \
  --win_size 32 \
  --decomp_kernel 15 \
  --period 25 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train

# 3. NIPS_TS_Water (Ref: Water)
# Best from search: win_size=32, decomp_kernel=55, period=55, k=3, lr=0.0001, epochs=3
echo ""
echo ">>> Running NIPS_TS_Water (Input: 9, Output: 9, Ratio: 0.5)..."
python main.py \
  --dataset NIPS_TS_Water \
  --data_path dataset/NIPS_TS_Water \
  --input_c 9 \
  --output_c 9 \
  --anormly_ratio 0.5 \
  --batch_size 256 \
  --win_size 32 \
  --decomp_kernel 55 \
  --period 55 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train

# 4. PSM
# Best from search: win_size=64, decomp_kernel=25, period=15, k=3, lr=0.0001, epochs=3
echo ""
echo ">>> Running PSM (Input: 25, Output: 25, Ratio: 1.0)..."
python main.py \
  --dataset PSM \
  --data_path dataset/PSM \
  --input_c 25 \
  --output_c 25 \
  --anormly_ratio 1.0 \
  --batch_size 256 \
  --win_size 64 \
  --decomp_kernel 25 \
  --period 15 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train

# 5. SWAT
# Best from search: win_size=64, decomp_kernel=25, period=55, k=3, lr=0.0001, epochs=3
echo ""
echo ">>> Running SWAT (Input: 51, Output: 51, Ratio: 0.1)..."
python main.py \
  --dataset SWAT \
  --data_path dataset/SWAT \
  --input_c 51 \
  --output_c 51 \
  --anormly_ratio 0.1 \
  --batch_size 256 \
  --win_size 64 \
  --decomp_kernel 25 \
  --period 55 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train

# 6. WADI
# Best from search: win_size=110, decomp_kernel=55, period=55, k=3, lr=0.0001, epochs=3
# Note: Batch size set to 128 to avoid OOM due to high channel count (127)
echo ""
echo ">>> Running WADI (Input: 127, Output: 127, Ratio: 0.1)..."
python main.py \
  --dataset WADI \
  --data_path dataset/WADI \
  --input_c 127 \
  --output_c 127 \
  --anormly_ratio 0.1 \
  --batch_size 128 \
  --win_size 110 \
  --decomp_kernel 55 \
  --period 55 \
  --k 3 \
  --lr 0.0001 \
  --num_epochs 3 \
  --mode train

echo "======================================================================="
echo "All runs completed."
echo "======================================================================="
