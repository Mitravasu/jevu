---
name: rl-evaluation-logger
description: Record JevU RL evaluation results as experiment iterations and keep the experiment summary current. Use when the user provides a new eval, asks to log or compare an RL run, or wants the iteration history updated. Do not use for running training or evaluation unless the user also requests it.
---

# RL Evaluation Logger

Maintain the experiment record under `experiment_log/` from evidence in evaluation JSON, bundle metadata, training metrics, simulation logs, and user-provided results.

## Files

- `experiment_log/iterations/NN-description.md`: one file per meaningful trained-policy iteration.
- `experiment_log/rl_experiment_summary.md`: linked overview and cross-iteration comparison.
- `experiment_log/dump.md`: historical source material; treat it as read-only unless the user asks otherwise.

Read [references/iteration-format.md](references/iteration-format.md) before creating a new iteration or substantially restructuring an existing one.

## Determine the target iteration

Inspect the evaluation's bundle path and the existing iteration notes before editing.

- If the bundle already has an iteration note, add the evaluation to that note. More seeds, sizes, or stress tests of the same saved policy are additional evidence, not new model versions.
- Create a new numbered iteration for a newly trained bundle, resumed or fine-tuned child model, materially changed reward/configuration, observation schema, algorithm, or environment distribution.
- If an existing proposal note describes the experiment that was actually run, complete that proposal note rather than creating a duplicate. Preserve its proposed plan and add the actual configuration and results.
- If the new experiment differs materially from a proposal, leave the proposal intact and create the next available iteration number.
- Smoke runs that only prove plumbing belong in the pipeline iteration unless they evaluate a meaningful trained policy.

Never infer model lineage from timestamps alone. Prefer `metadata.json` lineage; otherwise state that lineage is unknown or comes from the user's description.

## Gather evidence

When paths are available, inspect:

1. The evaluation JSON for suite settings, per-case results, and aggregates.
2. The evaluated bundle's `metadata.json` for schema, environment, PPO settings, device, requested/actual timesteps, and parent lineage.
3. `training_metrics.jsonl` only when a learning-curve statement would be useful.
4. A referenced simulation JSONL only when explaining observed behavior.
5. The parent iteration and evaluation when making a before/after comparison.

Prefer computed values from artifacts over values copied from console text. If only pasted results are available, record them and label the source as user-provided. Do not invent missing paths, settings, or metrics.

## Compare correctly

- Compare aggregate scores directly only when evaluation sizes, seeds, agent count, turn limit, tree density, and scoring weights match.
- When suites differ, report each result but mark the comparison as directional or non-equivalent.
- Separate overall results from the in-distribution training size when that distinction matters.
- Note duplicate seeds, partial runs, changed worker counts, changed rollout sizes, and other confounders.
- Treat PPO's rounded actual timestep count separately from requested timesteps.
- Do not treat training loss as a direct performance metric.

## Update the records

For a new iteration:

1. Choose the next two-digit number from the existing filenames.
2. Use a short descriptive kebab-case filename.
3. Write the evidence, results, interpretation, and the improvement that led to or follows from the iteration.
4. Add a relative link near the top of `rl_experiment_summary.md`.
5. Add or update the at-a-glance row and relevant comparison/conclusion text in the summary.

For another evaluation of an existing iteration:

1. Add an `Additional evaluation` subsection with the evaluation path and suite settings.
2. Record both aggregate and important per-size results.
3. Update the iteration's interpretation only when the new evidence changes confidence or conclusions.
4. Update summary numbers only if this evaluation becomes the declared primary baseline; otherwise mention it as supporting or stress-test evidence.

Preserve prior results. Correct a prior value only when the artifact proves it was wrong, and explain the correction briefly in the note.

## Handoff

Report which iteration file was created or updated, which summary entries changed, the headline result, and any comparability limitation. Do not claim a new “best” model unless the relevant evaluation is comparable to the current baseline.
