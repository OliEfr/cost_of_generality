"""G3 dataset QA over the generated HDF5 pools: visual grid, action ranges,
randomization coverage (cup init XY, goal XY), final placement error."""
import argparse, glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def episodes(files):
    for path in files:
        with h5py.File(path, "r") as f:
            for demo in f["data"]:
                yield path, demo, f["data"][demo]


def qa_level(name, files, outdir):
    cup0, goal, cupT, lengths = [], [], [], []
    amin = np.full(7, np.inf)
    amax = np.full(7, -np.inf)
    frames = []
    n_eps = 0
    for path in files:
        with h5py.File(path, "r") as f:
            n_eps += len(f["data"])
    grid_idx = set(np.linspace(0, n_eps - 1, 16, dtype=int).tolist())
    for i, (path, demo, g) in enumerate(episodes(files)):
        cup0.append(g["initial_state/rigid_object/cup/root_pose"][0, :2])
        goal.append(g["initial_state/rigid_object/goal_marker/root_pose"][0, :2])
        cupT.append(g["obs/cup_pos"][-1, :2])
        a = g["actions"][:]
        lengths.append(a.shape[0])
        amin = np.minimum(amin, a.min(0))
        amax = np.maximum(amax, a.max(0))
        if i in grid_idx:
            frames.append(g["obs/table_cam"][0])
    cup0, goal, cupT = map(np.array, (cup0, goal, cupT))
    lengths = np.array(lengths)
    place_err = np.linalg.norm(cupT - goal, axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, im in zip(axes.flat, frames):
        ax.imshow(im); ax.axis("off")
    fig.suptitle(f"{name}: first table_cam frame, 16 episodes")
    fig.tight_layout(); fig.savefig(f"{outdir}/{name}_grid.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(cup0[:, 0], cup0[:, 1], s=6, label="cup init")
    ax.scatter(goal[:, 0], goal[:, 1], s=6, label="goal")
    ax.scatter(cupT[:, 0], cupT[:, 1], s=6, marker="x", label="cup final")
    ax.set_aspect("equal"); ax.legend(); ax.set_title(f"{name} coverage (XY, world)")
    fig.savefig(f"{outdir}/{name}_coverage.png", dpi=110); plt.close(fig)

    def box(p):
        return (f"x[{p[:,0].min():+.3f},{p[:,0].max():+.3f}] "
                f"y[{p[:,1].min():+.3f},{p[:,1].max():+.3f}] "
                f"span {p[:,0].ptp()*100:.1f}x{p[:,1].ptp()*100:.1f} cm")
    print(f"== {name}: {n_eps} eps, len min/med/max {lengths.min()}/{int(np.median(lengths))}/{lengths.max()}")
    print(f"   cup init {box(cup0)}")
    print(f"   goal     {box(goal)}")
    print(f"   final |cup-goal| xy: med {np.median(place_err)*100:.2f} cm, p95 {np.percentile(place_err,95)*100:.2f} cm, max {place_err.max()*100:.2f} cm (success gate 5 cm)")
    print(f"   action min {np.round(amin,3)}")
    print(f"   action max {np.round(amax,3)}")
    assert place_err.max() < 0.05 + 1e-6, f"{name}: an episode ends outside the success radius"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="ops/qa")
    args = ap.parse_args()
    levels = {L: [f"data/hdf5/{L}.hdf5"] for L in ("L0", "L1", "L2")}
    levels["L3"] = sorted(set(glob.glob("data/hdf5/L3v0[0-9].hdf5")) - set(glob.glob("*_failed*")))
    for name, files in levels.items():
        qa_level(name, files, args.outdir)
    print("QA_DONE")
