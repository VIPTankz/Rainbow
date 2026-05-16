# Rainbow DQN

A clean implementation of [Rainbow DQN](https://arxiv.org/abs/1710.02298) for Atari, intended for upcoming integration into Stable Baselines.

Includes all six Rainbow components:
- Double DQN
- Dueling networks
- Prioritized Experience Replay (PER)
- N-step returns
- Distributional RL (C51)
- Noisy Nets

## Installation

Requires Python 3.11 and a CUDA-capable GPU. Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
# with uv
uv sync

# or with pip
pip install -e .
```

This installs PyTorch (CUDA 12.8 wheels), Gymnasium 0.29.1, ALE-py and the Atari ROMs via `autorom`.

## Running

The training script is `Rainbow.py`. The only required setup is picking a game; everything else has defaults matching the Rainbow paper.

```bash
python Rainbow.py --game NameThisGame
```

### Common arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--game` | `NameThisGame` | Atari game ID (anything under `ALE/<game>-v5`) |
| `--frames` | `200000000` | Total environment frames (frames / 4 = agent steps) |
| `--envs` | `64` | Parallel environments |
| `--bs` | `32` | Batch size |
| `--nstep` | `3` | N for n-step returns |
| `--lr` | `6.25e-5` | Learning rate |
| `--spi` | `16` | Samples per insert (replay ratio) |
| `--target_replace_frames` | `32000` | Target network update period (in frames) |
| `--discount` | `0.99` | Gamma |
| `--per_alpha` | `0.5` | Priority exponent for PER |
| `--linear_size` | `512` | Width of the FC layers |
| `--framestack` | `4` | Number of stacked frames |
| `--sticky` | `1` | Sticky actions (repeat prob 0.25) |
| `--repeat` | `0` | Run index — use to launch multiple seeds |
| `--include_evals` | `1` | Run the periodic evaluation protocol |
| `--num_eval_episodes` | `20` | Episodes per evaluation |
| `--eval_envs` | `5` | Parallel envs used for evaluation |
| `--testing` | `False` | Tiny config for local debugging |

### Outputs

Each run creates a directory named `Rainbow_<game><frames>M[_<non-default args>]` containing:
- `<agent_name>Experiment.npy` — training scores `[score, step]`
- `<agent_name>Evaluation.npy` — evaluation scores, one row per 250k step checkpoint
- `<agent_name>_<n>M.model` — periodic model checkpoints

## Running on HPC (SLURM)

`execute_jobs.sh` sweeps a list of games (and seeds) across several SLURM partitions. Before submitting:

1. Confirm the partitions in `partitions=(...)` exist on your cluster.
2. Confirm the conda environment name (`Rainbow` by default) matches what you created.
3. Clone the repo to `~/Rainbow` on the cluster (or edit the `cd` path in the script).

Then submit:

```bash
bash execute_jobs.sh
```
