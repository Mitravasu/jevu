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
