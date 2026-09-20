# Iteration note format

Use only the sections supported by the available evidence. Do not add empty headings.

```markdown
# Iteration NN: descriptive name

## Bundle

`artifacts/rl/.../RUN_DIRECTORY`

Parent: `...` when verified.

## Change from the previous iteration

What was intentionally changed and why.

## Configuration

- Requested and actual timesteps.
- Observation schema and algorithm variant.
- Agent count, training world distribution, and parallel workers.
- `n_steps`, rollout size, or reward values when relevant.

## Primary evaluation

Evaluation: `path/to/evaluation.json`

Suite: sizes, seeds, agents, max turns, density, and scoring weights.

| Size | Score | Survival | Coverage |
|---|---:|---:|---:|
| ... | ... | ... | ... |
| **Overall** | **...** | **...** | **...** |

## Comparison

A compact table against the most relevant comparable predecessor.

## Training trend

Only include evidence derived from `training_metrics.jsonl`.

## Interpretation

What the result demonstrates, likely limitations, and important confounders.

## Improvement for the next iteration

Record an implemented next change as fact. Label unimplemented ideas as proposed.
```

## Multiple evaluations of one bundle

Keep the primary evaluation unchanged unless the user explicitly promotes a new suite or the new run is a strict replacement. Add:

```markdown
## Additional evaluation: short label

Evaluation: `path`

- Suite differences.
- Aggregate results.
- Per-size or per-seed findings that change the interpretation.
- Comparability caveat.
```

## Summary entry

The summary should link the note and answer four questions compactly:

1. What model/configuration was evaluated?
2. What was the headline result?
3. What changed relative to the prior iteration?
4. What improvement or open question followed?

Keep detailed evidence in the iteration note rather than duplicating it all in the summary.
