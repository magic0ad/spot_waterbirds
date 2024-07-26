#!/bin/bash
#SBATCH --job-name=sn6_spot_waterbird_student
#SBATCH --output=/home/mila/j/jaewoo.lee/logs/spot_waterbird_student_sn6%j.out
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

python train_spot_2.py --dataset waterbird --data_path /home/mila/j/jaewoo.lee/scratch/dataset/waterbirds --epochs 560 --num_slots 6 --train_permutations random --eval_permutations standard --teacher_train_permutations random --teacher_eval_permutations random --teacher_checkpoint_path /home/mila/j/jaewoo.lee/scratch/result/spot_teacher_waterbird/2024-07-09T14:40:35.447825/checkpoint.pt.tar --log_path ~/scratch/result/spot_student_waterbird_sn6
