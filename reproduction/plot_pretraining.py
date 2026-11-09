"""Report progress of the reproduction run and plot its loss curve.

Run from the repository root:  python reproduction/plot_pretraining.py [WORK_DIR]
Reads TensorBoard logs written by GPT2_model.py (WORK_DIR/runs/) and, if present,
WORK_DIR/eval_results.json from eval_checkpoints.py. Prints a flatness summary and
writes results/pretraining_loss_light.png and results/pretraining_loss_dark.png.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORK = sys.argv[1] if len(sys.argv) > 1 else "reproduction/work"
TOKENS_PER_STEP = 25 * 1024  # batch size x context, as in GPT2_model.py
SECONDS_PER_STEP_HINT = None

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  train="#2a78d6", val="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#383835",
                 train="#3987e5", val="#d95926"),
}


def load_scalars():
    train, val = [], []
    for path in sorted(glob.glob(os.path.join(WORK, "runs", "*"))):
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if "Loss/train" in tags:
            train += [(e.step, e.value, e.wall_time) for e in ea.Scalars("Loss/train")]
        if "Loss/val" in tags:
            val += [(e.step, e.value, e.wall_time) for e in ea.Scalars("Loss/val")]
    return sorted(train), sorted(val)


def rolling(values, half_window=15):
    out = []
    for i in range(len(values)):
        w = values[max(0, i - half_window): i + half_window + 1]
        out.append(sum(w) / len(w))
    return out


def report(train, val):
    if not train:
        print("No training logs yet.")
        return
    step, _, t_last = train[-1]
    hours = (t_last - train[0][2]) / 3600
    print(f"step {step:,}  |  {step * TOKENS_PER_STEP / 1e9:.2f}B tokens  |  {hours:.1f} h elapsed")
    if len(train) > 1:
        rate = (train[-1][0] - train[0][0]) / max(t_last - train[0][2], 1)
        print(f"{rate:.2f} steps/s  ({rate * TOKENS_PER_STEP / 1000:.0f}k tokens/s)")
    if val:
        print("recent validation loss:", ", ".join(f"{v:.3f}" for _, v, _ in val[-6:]))
        # Change in validation loss over roughly the last 6 hours of training
        cutoff = val[-1][2] - 6 * 3600
        earlier = [v for _, v, t in val if t <= cutoff]
        if earlier:
            print(f"val loss change over last ~6 h: {val[-1][1] - earlier[-1]:+.4f}")


def plot(train, val, evals):
    ts = [s * TOKENS_PER_STEP / 1e9 for s, _, _ in train]
    tl = [v for _, v, _ in train]
    vs = [s * TOKENS_PER_STEP / 1e9 for s, _, _ in val]
    vl = [v for _, v, _ in val]

    for name, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        fig.patch.set_facecolor(c["surface"])
        ax.set_facecolor(c["surface"])

        ax.plot(ts, tl, color=c["train"], linewidth=0.6, alpha=0.25)
        ax.plot(ts, rolling(tl), color=c["train"], linewidth=2, label="Training loss (smoothed)")
        if val:
            ax.plot(vs, vl, color=c["val"], linewidth=2, label="Validation loss")
            ax.text(vs[-1], vl[-1] + 0.06, f"val {vl[-1]:.2f}", color=c["text"], fontsize=9, ha="right")

        # The original checkpoints, scored on a fixed sample of the same validation books.
        # Their x-position (tokens seen) is unknown, so they are drawn as reference levels.
        for key, label in [("original_final", "original final.pt"), ("original_latest", "original latest.pt")]:
            if key in evals:
                y = evals[key]["val_loss"]
                ax.axhline(y, color=c["muted"], linewidth=1, linestyle=(0, (3, 3)))
                ax.text(0.01, y + 0.03, f"{label} (step {evals[key]['global_step']:,}): {y:.2f}",
                        transform=ax.get_yaxis_transform(), color=c["muted"], fontsize=9)

        refs = [evals[k]["val_loss"] for k in ("original_final", "original_latest") if k in evals]
        lo = max(0, min(tl + vl + refs) - 0.3)
        hi = 6.0 if min(tl) < 4 else max(tl) + 0.2  # clip the first few steps (loss ~9.5)
        ax.set_ylim(lo, hi)
        if tl[0] > hi:
            ax.text(0.01, 0.97, f"starts at {tl[0]:.1f}", transform=ax.transAxes,
                    color=c["muted"], fontsize=9, va="top")
        ax.set_xlim(0, ts[-1])
        ax.set_title("Reproduction: pretraining GPT-2 (95M) on Project Gutenberg", color=c["text"],
                     loc="left", fontsize=12, pad=12)
        ax.set_xlabel("Training tokens seen (billions)", color=c["muted"])
        ax.set_ylabel("Cross-entropy loss", color=c["muted"])
        ax.tick_params(colors=c["muted"], length=0)
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper right", frameon=False, labelcolor=c["text"])
        fig.tight_layout()
        os.makedirs("results", exist_ok=True)
        fig.savefig(f"results/pretraining_loss_{name}.png", facecolor=c["surface"])
        plt.close(fig)


def main():
    train, val = load_scalars()
    report(train, val)
    eval_path = os.path.join(WORK, "eval_results.json")
    evals = json.load(open(eval_path)) if os.path.exists(eval_path) else {}
    for k, v in evals.items():
        print(f"{k}: step {v['global_step']:,}  val loss {v['val_loss']:.3f}  ppl {v['perplexity']:.1f}")
    if train:
        plot(train, val, evals)


if __name__ == "__main__":
    main()
