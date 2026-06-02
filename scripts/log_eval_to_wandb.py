"""Log OOD eval results to W&B as tables, bar charts, and matplotlib line charts.

Run after modal_eval_ood.py has produced eval_ood_results.json:
    python scripts/log_eval_to_wandb.py
    python scripts/log_eval_to_wandb.py --results path/to/eval_ood_results.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import wandb

OOD_THICKNESS_TAGS = ["t75um", "t100um", "t200um", "t300um"]
OOD_HTC_TAGS = ["rconv0p01", "rconv0p05", "rconv0p20", "rconv0p50"]

CONDITION_LABELS = {
    "eval_id":   "In-dist",
    "t75um":     "75",
    "t100um":    "100",
    "t200um":    "200",
    "t300um":    "300",
    "rconv0p01": "0.01",
    "rconv0p05": "0.05",
    "rconv0p20": "0.20",
    "rconv0p50": "0.50",
}

# Color per architecture, line style per λ
ARCH_COLORS = {
    "unet":            "#2196F3",  # blue
    "plain_cnn":       "#F44336",  # red
    "encoder_decoder": "#4CAF50",  # green
}
LAM_STYLES = {
    "0.00": ("solid",   "o"),
    "0.01": ("dashed",  "s"),
    "0.10": ("dashdot", "^"),
    "1.00": ("dotted",  "D"),
}

def _parse_model(name: str):
    """Return (arch, lam_str) from display name like 'unet_lam0.01'."""
    if name.startswith("unet_lam"):
        return "unet", name.replace("unet_lam", "")
    if name.startswith("plain_cnn_lam"):
        return "plain_cnn", name.replace("plain_cnn_lam", "")
    if name.startswith("enc_dec_lam"):
        return "encoder_decoder", name.replace("enc_dec_lam", "")
    return "unet", "0.00"


def _make_line_fig(model_names, results, tags, xs, xlabel, ylabel, title, training_x=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name in model_names:
        arch, lam_str = _parse_model(name)
        color = ARCH_COLORS.get(arch, "#888888")
        style, marker = LAM_STYLES.get(lam_str, ("solid", "o"))
        ys = [
            results[name][t]["rmse_K"] if ylabel == "RMSE (K)" else results[name][t]["hotspot_iou"]
            if t in results[name] else np.nan
            for t in tags
        ]
        ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                linewidth=1.8, markersize=5, label=name)

    if training_x is not None:
        ax.axvline(training_x, color="gray", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"Training ({training_x})")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Two-part legend: architecture (color) + λ (style)
    arch_handles = [
        mlines.Line2D([], [], color=c, linewidth=2, label=a.replace("_", " ").title())
        for a, c in ARCH_COLORS.items()
    ]
    lam_handles = [
        mlines.Line2D([], [], color="black", linestyle=s, marker=mk,
                      markersize=5, linewidth=1.8, label=f"λ={l}")
        for l, (s, mk) in LAM_STYLES.items()
    ]
    leg1 = ax.legend(handles=arch_handles, loc="upper left", fontsize=9,
                     title="Architecture", framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=lam_handles, loc="upper right", fontsize=9,
              title="Physics λ", framealpha=0.8)

    plt.tight_layout()
    return fig


def _make_bar_fig(model_names, results, metric_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    vals, colors, labels = [], [], []
    for name in model_names:
        m = results[name].get("eval_id")
        if m:
            arch, _ = _parse_model(name)
            vals.append(m[metric_key])
            colors.append(ARCH_COLORS.get(arch, "#888"))
            labels.append(name)

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    arch_handles = [
        mlines.Line2D([], [], color=c, marker="s", markersize=10,
                      linestyle="None", label=a.replace("_", " ").title())
        for a, c in ARCH_COLORS.items()
    ]
    ax.legend(handles=arch_handles, fontsize=9, title="Architecture")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_ood_results.json")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    model_names = list(results.keys())
    print(f"Loaded results for {len(model_names)} models")

    run = wandb.init(
        project="gpu-thermal-prediction",
        name="ood-eval-summary",
        job_type="eval",
        config={"results_file": str(results_path), "n_models": len(model_names)},
    )

    # ── Full results table ───────────────────────────────────────────────────
    full_table = wandb.Table(columns=["model", "condition", "rmse_K", "ssim", "hotspot_iou", "n"])
    for model, cond_results in results.items():
        for cond, m in cond_results.items():
            full_table.add_data(
                model, CONDITION_LABELS.get(cond, cond),
                round(m["rmse_K"], 4), round(m["ssim"], 5),
                round(m["hotspot_iou"], 4), m["n"],
            )
    run.log({"results/full_table": full_table})

    # ── In-distribution bar charts ───────────────────────────────────────────
    fig_rmse = _make_bar_fig(model_names, results, "rmse_K",
                             "RMSE (K)", "In-Distribution RMSE by Model")
    fig_iou  = _make_bar_fig(model_names, results, "hotspot_iou",
                             "Hotspot IoU", "In-Distribution Hotspot IoU by Model")
    run.log({
        "in_dist/rmse_bar":        wandb.Image(fig_rmse),
        "in_dist/hotspot_iou_bar": wandb.Image(fig_iou),
    })
    plt.close("all")

    # ── OOD thickness line charts ────────────────────────────────────────────
    thickness_xs = [75, 100, 200, 300]
    fig_t_rmse = _make_line_fig(
        model_names, results, OOD_THICKNESS_TAGS, thickness_xs,
        "Die Thickness (µm)", "RMSE (K)",
        "RMSE vs Die Thickness (OOD)  [trained at 150 µm]",
        training_x=150,
    )
    fig_t_iou = _make_line_fig(
        model_names, results, OOD_THICKNESS_TAGS, thickness_xs,
        "Die Thickness (µm)", "hotspot_iou",
        "Hotspot IoU vs Die Thickness (OOD)  [trained at 150 µm]",
        training_x=150,
    )
    run.log({
        "ood_thickness/rmse": wandb.Image(fig_t_rmse),
        "ood_thickness/iou":  wandb.Image(fig_t_iou),
    })
    plt.close("all")

    # ── OOD HTC line charts ──────────────────────────────────────────────────
    htc_xs = [0.01, 0.05, 0.20, 0.50]
    fig_h_rmse = _make_line_fig(
        model_names, results, OOD_HTC_TAGS, htc_xs,
        "r_convec (K/W)", "RMSE (K)",
        "RMSE vs Convection Resistance (OOD)  [trained at 0.1 K/W]",
        training_x=0.1,
    )
    run.log({"ood_htc/rmse": wandb.Image(fig_h_rmse)})
    plt.close("all")

    # ── Summary scalars ──────────────────────────────────────────────────────
    summary_update = {}
    for model in model_names:
        m = results[model].get("eval_id")
        if m:
            safe = model.replace(".", "p")
            summary_update[f"{safe}/rmse_K"]      = m["rmse_K"]
            summary_update[f"{safe}/ssim"]        = m["ssim"]
            summary_update[f"{safe}/hotspot_iou"] = m["hotspot_iou"]
    run.summary.update(summary_update)

    run.finish()
    print(f"\nDone. View at: {run.url}")


if __name__ == "__main__":
    main()
