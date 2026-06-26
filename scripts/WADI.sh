export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.1 --num_epochs 3 --batch_size 128 --mode train --dataset WADI --data_path dataset/WADI --input_c 127 --output_c 127