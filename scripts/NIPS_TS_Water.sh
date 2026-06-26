export CUDA_VISIBLE_DEVICES=0

# NIPS_TS_Water (NPY, 9 channels)
python main.py --anormly_ratio 1 --num_epochs 2 --batch_size 256 --mode train --dataset NIPS_TS_Water --data_path dataset/NIPS_TS_Water --input_c 9 --output_c 9 --decomp_kernel 15 --period 15 --win_size 64
