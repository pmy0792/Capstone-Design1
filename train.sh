#!/bin/bash

#SBATCH --job-name=OURS_seed21_size10_topk5_4_2_1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w agi2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=8:00:00
#SBATCH -o %x_%j.out
#SBTACH -e %x_%j.err
conda activate l2p-pytorch
python -m torch.distributed.launch \
        --master_port=29513 \
        --nproc_per_node=1 \
        --use_env main.py \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5 \
        --size 10 \
        --top_k 5 \
        --seed 21 \
        --pruning True \
        --pull_constraint_coeff 0.2
python -m torch.distributed.launch \
        --master_port=29513 \
        --nproc_per_node=1 \
        --use_env main.py \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5 \
        --size 10 \
        --top_k 4 \
        --seed 21 \
        --pruning True \
        --pull_constraint_coeff 0.2
python -m torch.distributed.launch \
        --master_port=29513 \
        --nproc_per_node=1 \
        --use_env main.py \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5 \
        --size 10 \
        --top_k 2 \
        --seed 21 \
        --pruning True \
        --pull_constraint_coeff 0.2
python -m torch.distributed.launch \
        --master_port=29513 \
        --nproc_per_node=1 \
        --use_env main.py \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5 \
        --size 10 \
        --top_k 1 \
        --seed 21 \
        --pruning True \
        --pull_constraint_coeff 0.2
