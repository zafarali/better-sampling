#!/bin/bash
#SBATCH --time=2:30:00
#SBATCH --job-name=baseline_ma_na10_mc5k_rw5
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

CYCLES=15
MC_SAMPLES=5000
N_AGENTS=10
RW_TIME=50
RW_WIDTH=5
OUTFOLDER="./baseline_experiments/baseline_ma_na10_mc5k_rw5"
REWARDCLIP=-10

python ./baseline_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 0 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed0 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP -nagents $N_AGENTS --no_tensorboard &

python ./baseline_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 2 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed2 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP -nagents $N_AGENTS --no_tensorboard &

python ./baseline_experiments/rvi_cycle_investigation.py --cycles $CYCLES --rw_seed 7 --rw_time $RW_TIME --rw_width $RW_WIDTH \
-samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed7 --outfolder $OUTFOLDER \
-rewardclip $REWARDCLIP -nagents $N_AGENTS --no_tensorboard &

wait