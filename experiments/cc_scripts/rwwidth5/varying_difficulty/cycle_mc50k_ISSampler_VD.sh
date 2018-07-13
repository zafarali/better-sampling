#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=cycle_mc50k_ISSampler_VD
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

CYCLES=100
MC_SAMPLES=1000
RW_TIME=50
RW_WIDTH=5
OUTFOLDER="./difficulty_experiments/ISSampler_cycle_mc50k_rw5"
REWARDCLIP=-10

for ENDPOINT in {0..15..1}
do
    python ./cycle_experiments/baseline_cycle_investigation.py --cycles $CYCLES --rw_seed 0 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed0 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} --method ISSampler &
    python ./cycle_experiments/baseline_cycle_investigation.py --cycles $CYCLES --rw_seed 2 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES -name rwseed2 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} --method ISSampler &
    python ./cycle_experiments/baseline_cycle_investigation.py --cycles $CYCLES --rw_seed 7 --rw_time $RW_TIME --rw_width $RW_WIDTH \
    -samseed $SLURM_ARRAY_TASK_ID -s $MC_SAMPLES --name rwseed7 -end_ov -endpoint $ENDPOINT --outfolder ${OUTFOLDER}_${ENDPOINT} --method ISSampler &
done

wait