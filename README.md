# The Cost of Generality

What does generality cost in robot manipulation imitation learning, measured in
demonstrations? For each task we define generality levels (breadth of the
training/eval distribution); per level we train vision diffusion policies at
multiple demo counts N and measure rollout success rate, deriving data-cost
curves N*(s | level) = demos needed to reach success rate s.

- Sim: Isaac Sim 5.1.0 + Isaac Lab v2.3.0 (fresh checkout in `third_party/`, pinned)
- Demos: scripted state-machine experts + Isaac Lab Mimic (MimicGen)
- Policy: LeRobot diffusion policy (pinned, see `docs/PINS.md`), vision obs, 1 seed/run, fixed 80k steps
- Tasks: T1 cup->target (start), T2 drawer+stow, T3 push-to-target
- Compute: this workstation (sim, datagen, debug) + CINECA Leonardo (training; evals if the A100 render gate passes)

Full study plan: `docs/PLAN.md`. Version pins: `docs/PINS.md`. Lab journal:
`docs/journal.md`. Run registry: `experiments/registry.csv`.

## Environments
- `cog_isaac` (conda, py3.11): Isaac Sim + Isaac Lab + (if compatible) LeRobot -- sim, datagen, eval
- `cog_lerobot` (conda): LeRobot pinned -- conversion + training (mirrored on cluster)

## Layout
See `docs/PLAN.md` (repository section). `data/` and `third_party/` are gitignored;
everything reproducible lives in git.
