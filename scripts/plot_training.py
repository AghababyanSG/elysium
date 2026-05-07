"""Generate training plots from TensorBoard logs for SFT and RL runs."""

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(run_dirs, tag):
    """Merge scalar tag across multiple run dirs, offsetting steps for continuity."""
    all_steps, all_vals = [], []
    offset = 0
    prev_max = 0
    for d in run_dirs:
        ea = EventAccumulator(d)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        # If this run restarts from step <= prev_max, it's a fresh restart — add offset
        if len(all_steps) > 0 and steps[0] <= prev_max:
            offset = prev_max
        all_steps.extend(steps + offset)
        all_vals.extend(vals)
        prev_max = max(all_steps) if all_steps else 0
    if not all_steps:
        return None, None
    order = np.argsort(all_steps)
    return np.array(all_steps)[order], np.array(all_vals)[order]


def smooth(vals, weight=0.85):
    smoothed, last = [], vals[0]
    for v in vals:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def plot_metric(ax, steps, vals, label, color, ylabel, smooth_weight=None, log_scale=False):
    if steps is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(label)
        return
    if smooth_weight is not None:
        ax.plot(steps, vals, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(steps, smooth(vals, smooth_weight), color=color, linewidth=1.8)
    else:
        ax.plot(steps, vals, color=color, linewidth=1.2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("step", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    if log_scale and np.all(np.array(vals) > 0):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_sft_plots(out_dir):
    sft_runs = sorted(glob.glob("models/checkpoints/runs/*/"))
    # Filter to runs with actual data
    valid = []
    for d in sft_runs:
        ea = EventAccumulator(d); ea.Reload()
        if ea.Tags().get("scalars"):
            valid.append(d)

    if not valid:
        print("No SFT data found.")
        return

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("SFT Training", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    metrics = [
        ("train/loss",          "Train Loss",          "loss",      "steelblue",  False),
        ("eval/loss",           "Eval Loss",           "loss",      "coral",      False),
        ("train/grad_norm",     "Gradient Norm",       "grad norm", "seagreen",   True),
        ("train/learning_rate", "Learning Rate",       "lr",        "orchid",     False),
    ]

    for idx, (tag, title, ylabel, color, log_scale) in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        s, v = load_scalar(valid, tag)
        plot_metric(ax, s, v, title, color, ylabel, log_scale=log_scale)

    out = out_dir / "sft_training.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_rl_plots(out_dir):
    rl_runs = sorted(glob.glob("models/checkpoints/rl_final/runs/*/"))
    if not rl_runs:
        print("No RL data found.")
        return

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("RL (GRPO) Training", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

    metrics = [
        ("train/reward",                          "Reward (mean)",           "reward",      "steelblue",       False),
        ("train/rewards/visual_reward_fn/mean",   "SSIM Reward",             "ssim reward", "mediumseagreen",  False),
        ("train/rewards/visual_reward_fn/std",    "SSIM Reward Std",         "std",         "lightseagreen",   False),
        ("train/kl",                              "KL Divergence",           "KL",          "tomato",          True),
        ("train/loss",                            "Policy Loss",             "loss",        "orchid",          False),
        ("train/completion_length",               "Completion Length",       "tokens",      "slategray",       False),
        ("train/frac_reward_zero_std",            "Frac Zero-Std Rewards",   "fraction",    "indianred",       False),
    ]

    for idx, (tag, title, ylabel, color, log_scale) in enumerate(metrics):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        s, v = load_scalar(rl_runs, tag)
        plot_metric(ax, s, v, title, color, ylabel, smooth_weight=0.85, log_scale=log_scale)

    out = out_dir / "rl_training.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_reward_detail(out_dir):
    """Larger reward-focused plot for the RL run."""
    rl_runs = sorted(glob.glob("models/checkpoints/rl_final/runs/*/"))
    if not rl_runs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RL Reward Detail", fontsize=13, fontweight="bold")

    s, v = load_scalar(rl_runs, "train/reward")
    if s is not None:
        axes[0].plot(s, v, alpha=0.25, color="steelblue", linewidth=0.8)
        axes[0].plot(s, smooth(v, 0.9), color="steelblue", linewidth=2)
        axes[0].fill_between(s, smooth(v, 0.9), alpha=0.15, color="steelblue")
        axes[0].set_title("Mean Reward over Training", fontweight="bold")
        axes[0].set_xlabel("step"); axes[0].set_ylabel("reward")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    s_m, v_m = load_scalar(rl_runs, "train/rewards/visual_reward_fn/mean")
    s_s, v_s = load_scalar(rl_runs, "train/rewards/visual_reward_fn/std")
    if s_m is not None:
        sm_smooth = smooth(v_m, 0.9)
        axes[1].plot(s_m, v_m, alpha=0.25, color="seagreen", linewidth=0.8)
        axes[1].plot(s_m, sm_smooth, color="seagreen", linewidth=2, label="mean SSIM reward")
        if s_s is not None and len(v_s) == len(v_m):
            ss_smooth = smooth(v_s, 0.9)
            axes[1].fill_between(s_m, sm_smooth - ss_smooth, sm_smooth + ss_smooth,
                                  alpha=0.2, color="seagreen", label="±1 std")
        axes[1].set_title("Visual (SSIM) Reward", fontweight="bold")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("SSIM reward")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    out = out_dir / "rl_reward_detail.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    out_dir = Path("training_plots")
    out_dir.mkdir(exist_ok=True)
    os.chdir(Path(__file__).parent.parent)

    save_sft_plots(out_dir)
    save_rl_plots(out_dir)
    save_reward_detail(out_dir)
    print(f"\nAll plots saved to: {out_dir.resolve()}")
    print("Copy to laptop: scp <host>:~/elysium/training_plots/*.png .")


if __name__ == "__main__":
    main()
