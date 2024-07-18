#!/bin/bash
#SBATCH --job-name=spot_coco
#SBATCH --output=/home/mila/j/jaewoo.lee/logs/spot_coco%j.out
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128Gb
#SBATCH --partition=long
#SBATCH --mail-user=jaewoo.lee@mila.quebec
#SBATCH --mail-type=ALL

module load miniconda/3
conda activate spot

cd /home/mila/j/jaewoo.lee/projects/spot

python train_spot.py --dataset coco --data_path ~/scratch/dataset/coco --epoch 50 --num_slots 7 --train_permutations random --eval_permutations standard --log_path ~/scratch/result/spot_teacher_coco
