#!/bin/bash
#SBATCH --job-name=spot_coco
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24Gb
#SBATCH --partition=main
#SBATCH --mail-user=lachaseb@mila.quebec

TAG=sa_variant_sigmoid_constraint_identity
STAGE=teacher
LOG_PATH=$SCRATCH/spot/exp/$TAG/$STAGE

module load anaconda/3
conda activate spot

cd /home/mila/l/lachaseb/spot_waterbirds

python train_spot.py --dataset coco --data_path $SCRATCH/spot/data/COCO2017 --epochs 50 --num_slots 7 --train_permutations random --eval_permutations standard --log_path $LOG_PATH --use_wandb --tag $TAG --reg_type constraint_identity --sa_variant True --lr_main 1e-4
#python train_spot.py --dataset coco --data_path ~/scratch/dataset/coco --epoch 50 --num_slots 7 --train_permutations random --eval_permutations standard --log_path ~/scratch/result/spot_teacher_coco
