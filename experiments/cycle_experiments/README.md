# Experiments to check if training and testing should happen separately. 

The set up is as follows:

1. Train for `$MC_SAMPLES`.
2. Evaluate the trajectories from this.
3. Evaluat the cumulative trajectories samples so far from this.
4. Go to 1 $CYCLES number of times.

To run this code

```bash
python ./cycle_experiments/rvi_cycle_investigation.py --cycles 20 --rw_seed 7 \
-samseed 5 -s 1000 -name rwseed7 --outfolder ./
```

The baselines can be run:
```bash
python ./cycle_experiments/baseline_cycle_investigation.py --cycles 20 --rw_seed 7 \
-samseed 5 -s 1000 --name rwseed7 --outfolder ./ --method ISSampler 
```