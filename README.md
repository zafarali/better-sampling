# Reinforced Variational Inference for Stochastic Processes

## Abstract Goals
The objective of this project is to learn the reverse transition dynamics of a stochastic process conditioned on a terminal observation. We will explore recent connections made between variational inference and policy-based reinforcement learning in \citet{weber2015reinforced} and \citet{bachman2015data} termed as reinforced variational inference. To our knowledge, this will be the first empirical evaluation of the above method and will provide insight into which techniques are transferable between the fields. We will evaluate the method on two tasks with known posterior distributions: the random walk and the two window random walk. We will then evaluate the method on two other stochastic processes from epidemiology and \zafcomment{population genetics}.

## How to Set up

This project requires installing PyTorch 0.3.1 (See requirements.txt for a full list of modules needed)
Additionally, this project requires installing `pg_methods`, a new tool to make implementing algorithms based policy gradients / REINFORCE simpler

To install that project:

```bash
git clone https://github.com/zafarali/policy-gradient-methods.git
cd policy-gradient-methods
pip install -e .
```

you can now install this one:

```bash
git clone https://github.com/zafarali/better-sampling.git
cd better-sampling
pip install -r requirements.txt
pip install -e .
```

To test if everything is working you can run:

```bash
python -m pytest ./tests
```

## Code Organization

There are two important parts of this code base: Stochastic Processes and Samplers.

### Stochastic Processes

A stochastic process is a sequence of random variables that are generated according to some transition kernel. The `StochasticProcess` object here is an abstraction of this.
They implement:

1. `state_space`: the size of the random variable 
2. `action_space`: the number of possible transitions from one variable to the next
4. `transition_prob`: Returns the transition probability from one `state_t` to `state_tp1` 
5. `simulate`: Simulates a new realization of the stochastic process

We can then interact with the stochastic process via "agents" or "particles" that start at the final observation. To interact with the stochastic process we have:

1. `reset`: Resets the stochastic process to the observation
2. `reset_task`: Generates a new realization (via `simulate`)
3. `step`: Executes an action in the environment and moves the particle one step back in time.

Concretely, we have the following workflow:

```python
from rvi_sampling.stochastic_processes import MyStochasticProcess

env = MyStochasticProcess(*stochastic_process_configuations, seed=2018)
state = env.reset_task() # generates a new task
state_tp1, log_prob, done, info =  env.step(action)
...
env.reset() # Resets the environment (but not the task!)
```

The `log_prob` is the log probability of taking the `action` under the stochastic process.

#### Implemented Stochastic Processes

1. Discrete Random Walk
2. Two Window Discrete Random Walk
3. SIR Epidemiology model (Work in Progress)

### Samplers

The most important part of the code base are `Sampler` objects that can be found in `./rvi_sampling/samplers/`.
The purpose of the sampler is to estimate the posterior distribution. Each sampler you implement must take the hyperparameters you are interested in as the `__init__` arguments (along with seed) and must implement the `solve
 method so that they can be benchmarked against each other. The `Sampler` interacts with the `StochasticProcess` via the `solve` command.
 
```python
from rvi_sampling.samplers import MyAwesomeSampler

sampler = MyAwesomeSampler(hyperparameter1=1, hyperparameter2=2)
sampler.solve(env, mc_samples=5000) # runs the sampler and obtains 5k MC Samples
```

`Sampler.solve` returns a `SamplingResults` object for easy visualizations.

### Testing the algorithm

To perform a single evaluation of the algorithm you can look at [`experiments/rw_experiment.py`](./experiments/rw_experiment.py)
To run a large scale evaluation of the algorithm on a SLURM cluster (like ComputeCanada) you can use [`./experiments/cc_random_walk_expectation_analysis.py`](./experiments/cc_random_walk_expectation_analysis.py) 
which will create a SLURM script to run the above experiment on 50 different tasks, averaged over 5 seeds.
You can visualize the results using:
1. `visualize_proposal.py`: visualize the proposal distribution
2. `KL_violin.py` Compare the KL values for each algorithm.

