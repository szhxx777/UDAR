#!/bin/bash

# Set the visible GPU device. Update this value if you use another GPU.
export CUDA_VISIBLE_DEVICES=0

echo "==============================================="
echo "Running direct evaluation for all datasets"
echo "==============================================="

echo ""
echo ">>> Testing MSL"
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

echo ""
echo ">>> Testing NIPS_TS_Swan"
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
  --mode test

echo ""
echo ">>> Testing NIPS_TS_Water"
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
  --mode test

echo ""
echo ">>> Testing PSM"
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
  --mode test

echo ""
echo ">>> Testing SWAT"
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
  --mode test

echo ""
echo ">>> Testing WADI"
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
  --mode test

echo "==============================================="
echo "All evaluations completed."
echo "==============================================="
