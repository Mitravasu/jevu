# JevU

JevU is a small agent survival simulation. Jev receives each agent's observable
state and selects one interaction and one exploration action per turn.

```sh
export JEV_API_TOKEN="..."
make run
```

The default model is `jev-1.13.0`. Override it with
`make run ARGS="--model MODEL_NAME"`.

The simulation owns turn order and action application. `JevActionSelector` only
selects typed `TurnActions`; it does not mutate the game.

Gameplay tuning values such as the hunger range, per-turn hunger loss, harvest
amount, eating cost, and hunger restoration live together in `jevu/rules.py`.

Extract one agent's actions from a simulation log as plain-English sentences:

```sh
make agent-actions ARGS="logs/SIMULATION.jsonl A1"
```

## Train an RL agent

Install the optional reinforcement-learning dependencies, then train from TOML:

```sh
make setup-rl
make train-rl ARGS="--config configs/rl/survival_claim.toml"
```

Use `--smoke` while developing to run a single short PPO rollout. Training does not
use Jev or require `JEV_API_TOKEN`. Every run writes a unique, gitignored bundle below
`artifacts/rl/` containing the final model, bounded recovery checkpoints, resolved
configuration, version/schema metadata, episode metrics, TensorBoard logs, and a
post-save reload smoke test.

`algorithm.device = "auto"` prefers Apple Metal (`mps`) when available, then CUDA,
then CPU. Set it explicitly to `mps`, `cuda`, or `cpu` to require that backend; an
unavailable explicitly requested accelerator fails before training starts. The
resolved device is recorded in `metadata.json`.

### Parallel training worlds

Training can collect experience from several independent worlds concurrently:

```toml
[algorithm]
parallel_envs = 4
```

Each world runs in its own spawned CPU process and contains the configured number of
agents. All worlds feed experience into one shared PPO policy, whose neural-network
updates still run on the configured MPS, CUDA, or CPU device. `parallel_envs = 1`
uses a local single-process environment.

`n_steps` is measured per world. The combined PPO rollout contains
`n_steps * parallel_envs` agent decisions, and `batch_size` must divide that combined
number. For example, `n_steps = 512` and `parallel_envs = 4` collect 2,048 decisions
before each PPO update. `total_timesteps` remains the total across every world, not a
per-world target. Four workers therefore collect a 90,000-timestep run collectively;
they do not collect 90,000 timesteps each.

### Shared-policy multi-agent training

Set `agent_count` above one to train the same policy through multiple agents competing
in one shared world:

```toml
[environment]
agent_count = 10
```

Agents act sequentially in their normal simulation order. Each action is one training
timestep and updates the world immediately, so later agents observe claims made by
earlier agents and cannot claim those tiles. Every agent contributes experience to
the same policy; the trainer does not create a separate neural network per agent.
PPO's advantage calculation follows each identity across the round—for example,
`A1(turn 3) -> A1(turn 4)`—instead of treating A2's state as the result of A1's
action. An agent's death terminates that trajectory once; dead agents produce no
placeholder transitions. At a rollout boundary, the one unmatched tail transition
per living agent is treated as a boundary because its next personal observation has
not been produced yet.

The RL observation is a relative one-tile view. Adjacent-agent occupancy is visible,
but moves into occupied tiles are deliberately not action-masked: the normal failed
move and ineffective-action penalty teach the policy how to react to collisions.

Consequently, `total_timesteps = 20000` means 20,000 total agent decisions. With ten
living agents that is approximately 2,000 shared-world turns, not 20,000 turns per
agent. As agents die, later turns contribute fewer decisions.

Training uses a single filling progress bar by default instead of Stable-Baselines'
metric tables. After the first PPO update, the bar also shows the latest overall
training loss. Configure the console display in TOML:

```toml
[run]
console_output = "progress" # progress, table, or quiet
```

You can override it for one run without editing the file:

```sh
make train-rl ARGS="--config configs/rl/survival_claim.toml --console-output quiet"
```

## Run a trained RL model

The training command prints its model bundle directory when it finishes. Pass that
directory to `run-rl` to use the trained policy instead of Jev:

```sh
make run-rl ARGS="--bundle artifacts/rl/survival-claim-v1/RUN_DIRECTORY --max-turns 250 --seed 42"
```

This path does not call Jev and does not require `JEV_API_TOKEN`. Add `--pygame` to
watch the trained policy:

```sh
make run-rl ARGS="--bundle artifacts/rl/survival-claim-v1/RUN_DIRECTORY --max-turns 250 --seed 42 --pygame"
```

The policy observes a fixed 3 by 3 agent-centered view: its current tile, the four
orthogonally adjacent tiles, visible boundaries, and any adjacent agents. It does not
receive absolute coordinates or the full map, so the same bundle can run on any world
size supported by the simulator. Bundle loading checks the game rules, action mapping,
and observation schema, and old global-map bundles fail clearly as incompatible. RL
inference uses the same automatic MPS/CUDA/CPU selection as training; pass `--device
cpu` if you want to override it.

For a manual comparison, run Jev and RL with the same world parameters:

```sh
make run ARGS="--width 10 --height 10 --tree-density 0.15 --seed 42 --agents 1 --max-turns 250"
make run-rl ARGS="--bundle artifacts/rl/survival-claim-v1/RUN_DIRECTORY --width 10 --height 10 --tree-density 0.15 --seed 42 --agents 1 --max-turns 250"
```

Both backends use the same simulation transition code. One pair of runs should not be
treated as a reliable performance result.

## Evaluate a trained RL model

Run a deterministic benchmark over five fixed seeds and supported square world
sizes from 5 through 25:

```sh
make eval-rl ARGS="--bundle artifacts/rl/survival-claim-v1/RUN_DIRECTORY"
```

The default suite tests 5, 10, 15, 20, and 25 square worlds. The egocentric policy
has no training-canvas size limit. The default agent count, turn limit, and tree
density come from the bundle's training
configuration. Override the matrix when needed:

```sh
make eval-rl ARGS="--bundle artifacts/rl/.../RUN_DIRECTORY --sizes 10 15 25 --seeds 11 22 33 --agents 10 --max-turns 250"
```

Each episode reports surviving agents, total agent-turns survived, and tiles claimed.
The 0–100 score gives equal weight to the normalized survival ratio and claimed-map
coverage:

```text
score = 100 * (survived agent-turn ratio + claimed tile ratio) / 2
```

Use `--survival-weight` and `--claim-weight` to change that balance. Results are also
saved under the bundle's `evaluations/` directory as JSON, including every episode and
the overall averages. These seeds are evaluation inputs only; they do not update the
model.

You can also load a completed bundle from Python through the same selector interface
used by Jev:

```python
from jevu.rl.selector import RLActionSelector

selector = RLActionSelector.from_bundle("artifacts/rl/.../RUN_DIRECTORY")
final_state = game_state.run(max_turns=250, action_selector=selector)
```
