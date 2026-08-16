# Agent guardrails -- cost_of_generality

1. **Strict no-delete outside this repo.** Never delete/modify anything outside
   `/home/admin_07/cost_of_generality` (and its cluster mirror `$WORK/cog`).
   Disk full -> ask the user to clean up; never make room yourself.
2. **Never touch the foreign eval job** (policy server, PID 1796345 or successors
   under `project_repos/vpro_mimicgen_eval`). Check `nvidia-smi` headroom before
   launching GPU work; cap `--num_envs` and concurrent jobs accordingly.
3. **Do not modify** the existing `isaaclab` conda env, `project_repos/isaac_lab/`,
   or any other project on this machine or on the cluster.
4. Cluster: account `-A euhpc_b38_106` explicitly on every job (default account is
   the expired B34). Compute nodes have no internet: `WANDB_MODE=offline`,
   `HF_HUB_OFFLINE=1`. Nothing on `$SCRATCH` that matters (40-day purge).
5. **Keep the docs current IN THE SAME WORK SESSION — never batch documentation.**
   Every version pin -> `docs/PINS.md` (with reason). Every run -> a row in
   `experiments/registry.csv`. Every finding, fix, gotcha, or gate result ->
   `docs/journal.md` (dated) at the moment it lands. Every design decision ->
   `docs/decisions.md` (ADR-style, with why + open VERIFY items). A finding that
   lives only in a conversation or a code comment is considered LOST.
6. Training: fixed 80k steps, identical hyperparams for all cells
   (`configs/train/diffusion_base.yaml` is frozen after G5a). One seed (0).
7. Eval sets under `configs/eval_sets/` are frozen benchmarks -- never regenerate.
