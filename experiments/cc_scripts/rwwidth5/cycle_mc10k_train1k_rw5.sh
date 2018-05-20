#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=cycle_mc10k_train1k_rw5
#SBATCH --ntasks=3
#SBATCH -o ./out_%x.txt
#SBATCH -e ./err_%x.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-4

cd $SLURM_SUBMIT_DIR

source $RVI_ENV

module load cuda/8.0.44
module load cudnn/7.0
module load qt

CYCLES=25
MC_SAMPLES=10000
TRAIN_STEPS=1000
RW_TIME=50
RW_WIDTH=5
OUTFOLDER="./cycle_mc10k_train1k_rw5"
REWARDCLIP=-10

python ./cycle_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 0 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES --train_steps $TRAIN_STEPS -name rwseed0 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP &
python ./cycle_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 2 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES --train_steps $TRAIN_STEPS -name rwseed2 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP &
python ./cycle_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 7 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES --train_steps $TRAIN_STEPS -name rwseed7 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP &
wait
