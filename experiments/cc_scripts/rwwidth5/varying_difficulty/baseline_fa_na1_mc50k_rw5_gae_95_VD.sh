#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=baseline_fa_na1_mc50k_rw5_gae_95_VD
#SBATCH --ntasks=1
#SBATCH -o ./out_%x.txt
#SBATCH -e ./err_%x.txt
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=3
#SBATCH --array=0-4

cd $SLURM_SUBMIT_DIR

source $RVI_ENV

module load cuda/8.0.44
module load cudnn/7.0
module load qt

CYCLES=50
MC_SAMPLES=1000
N_AGENTS=1
RW_TIME=50
RW_WIDTH=5
OUTFOLDER="./difficulty_experiments/baseline_fa_na1_mc50k_rw5_gae_95"
REWARDCLIP=-10
BASELINE_LEARNING_RATE="1e-5"
LAM=0.95

for ENDPOINT in {0..20..2}
do
    python ./baseline_experiments/rvi_baseline_investigation.py --cycles $CYCLES --rw_seed 0 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed0 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} \
    -rewardclip $REWARDCLIP -baseline fn_approximation -baseline_nn 2 2 2 -baseline_lr $BASELINE_LEARNING_RATE -nagents $N_AGENTS \
    --no_tensorboard -gae -lam $LAM &

    python ./baseline_experiments/rvi_baseline_investigation.py --cycles $CYCLES --rw_seed 2 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed2 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} \
    -rewardclip $REWARDCLIP -baseline fn_approximation -baseline_nn 2 2 2 -baseline_lr $BASELINE_LEARNING_RATE -nagents $N_AGENTS \
    --no_tensorboard -gae -lam $LAM &

    python ./baseline_experiments/rvi_baseline_investigation.py --cycles $CYCLES --rw_seed 7 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed7 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} \
    -rewardclip $REWARDCLIP -baseline fn_approximation -baseline_nn 2 2 2 -baseline_lr $BASELINE_LEARNING_RATE -nagents $N_AGENTS \
    --no_tensorboard -gae -lam $LAM &
done

wait