#!/bin/bash
#SBATCH --job-name=sn3_spot_waterbird
#SBATCH --output=/home/mila/j/jaewoo.lee/logs/spot_waterbird_sn3_%j.out
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32Gb
#SBATCH --partition=long
#SBATCH --mail-user=jaewoo.lee@mila.quebec
#SBATCH --mail-type=ALL

module load miniconda/3
conda activate spot

cd /home/mila/j/jaewoo.lee/projects/spot

python train_spot.py --dataset waterbird --data_path ~/scratch/dataset/waterbirds --epoch 560 --num_slots 3 --train_permutations random --eval_permutations standard --log_path ~/scratch/result/spot_teacher_waterbird
