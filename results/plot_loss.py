"""Plot the QLoRA fine-tuning loss curve from the Trainer's saved log history.

Run from the repository root:  python results/plot_loss.py
Reads sft-neo-qlora/checkpoint-39003/trainer_state.json (not in the repository)
and writes results/qlora_loss_light.png and results/qlora_loss_dark.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE = "sft-neo-qlora/checkpoint-39003/trainer_state.json"

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  train="#2a78d6", eval="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#383835",
                 train="#3987e5", eval="#d95926"),
}


def ema(values, half_window=25):
    """Centred rolling mean (no lag, unlike an exponential average)."""
    out = []
    for i in range(len(values)):
        window = values[max(0, i - half_window): i + half_window + 1]
        out.append(sum(window) / len(window))
    return out


def main():
    state = json.load(open(STATE))
    history = state["log_history"]
    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    steps_per_epoch = state["max_steps"] / state["num_train_epochs"]

    ts, tl = zip(*train)
    es, el = zip(*evals)

    for name, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        fig.patch.set_facecolor(c["surface"])
        ax.set_facecolor(c["surface"])

        ax.plot(ts, tl, color=c["train"], linewidth=0.6, alpha=0.25)
        ax.plot(ts, ema(list(tl)), color=c["train"], linewidth=2, label="Training loss (smoothed)")
        ax.plot(es, el, color=c["eval"], linewidth=2, label="Eval loss (64 training examples)")

        for e in range(1, int(state["num_train_epochs"])):
            ax.axvline(e * steps_per_epoch, color=c["muted"], linewidth=1, linestyle=(0, (3, 3)))
            ax.text(e * steps_per_epoch, 0.98, f" epoch {e + 1}", transform=ax.get_xaxis_transform(),
                    color=c["muted"], fontsize=9, va="top")

        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, ts[-1])
        ax.annotate(f"starts at {tl[0]:.1f}", xy=(ts[0], 1.0), xytext=(600, 0.9),
                    color=c["muted"], fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=c["muted"], linewidth=1))
        # Direct labels at the right end
        ax.text(ts[-1] * 0.995, ema(list(tl))[-1] - 0.05, f"train {ema(list(tl))[-1]:.2f}",
                color=c["text"], fontsize=9, ha="right")
        ax.text(es[-1] * 0.995, el[-1] + 0.03, f"eval {el[-1]:.2f}", color=c["text"], fontsize=9, ha="right")

        ax.set_title("QLoRA fine-tune of GPT-Neo 1.3B on Alpaca", color=c["text"], loc="left",
                     fontsize=12, pad=12)
        ax.set_xlabel("Optimizer step", color=c["muted"])
        ax.set_ylabel("Cross-entropy loss", color=c["muted"])
        ax.tick_params(colors=c["muted"], length=0)
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        leg = ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), frameon=False, labelcolor=c["text"])

        fig.tight_layout()
        fig.savefig(f"results/qlora_loss_{name}.png", facecolor=c["surface"])
        plt.close(fig)
    print("train", tl[0], "->", tl[-1], "| eval", el[0], "->", el[-1], "| steps/epoch", steps_per_epoch)


if __name__ == "__main__":
    main()
