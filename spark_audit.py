"""
=============================================================
 A3L Big Data Pipeline — spark_audit.py
 Course: CS4074 Big Data

 WHAT THIS FILE DOES:
   Stage 7 — Monitoring & Audit Layer
   Logs pipeline behavior, anomaly counts, timing, and
   ML performance metrics to a structured audit report.
   Generates matplotlib charts for the project report.
=============================================================
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside a VM
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def banner(title: str):
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


# ─────────────────────────────────────────────
#  AUDIT LOG
# ─────────────────────────────────────────────

class PipelineAuditLogger:
    """
    Structured logger for the A3L Big Data Pipeline.
    Records every stage's input/output counts, timing, and
    anomaly statistics.  Saves everything to a JSON audit file
    and prints a human-readable summary.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log: Dict[str, Any] = {
            "pipeline": "A3L CTI Big Data Pipeline",
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

    def record_stage(self, stage: str, data: dict):
        self.log["stages"][stage] = {**data, "recorded_at": datetime.now().isoformat()}
        print(f"  📋  Audit recorded : {stage}")

    def save(self) -> str:
        path = os.path.join(self.output_dir, "audit_log.json")
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"  ✅  Audit log saved → {path}")
        return path

    def print_summary(self):
        banner("AUDIT SUMMARY")
        for stage, info in self.log["stages"].items():
            print(f"\n  [{stage}]")
            for k, v in info.items():
                if k != "recorded_at":
                    print(f"    {k:<30}: {v}")


# ─────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_label_distribution(stats: dict, output_dir: str):
    """Bar chart of label distribution in training data."""
    train_stats = stats.get("train", {})
    label_dist  = train_stats.get("label_distribution", {})
    if not label_dist:
        return

    labels = list(label_dist.keys())
    counts = list(label_dist.values())

    # Sort descending
    pairs  = sorted(zip(counts, labels), reverse=True)
    counts, labels = zip(*pairs)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, counts, color=colors)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)

    ax.set_xlabel("Number of Records", fontsize=12)
    ax.set_title("Label Distribution in Training Data (after ETL & Adaptation)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, "label_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def plot_anomaly_summary(flagged_counts: dict, output_dir: str):
    """
    Stacked bar chart: clean vs anomaly records per split.
    Shows the adversarial detection stage results visually.
    """
    if not flagged_counts:
        return

    splits = list(flagged_counts.keys())
    clean_counts   = [flagged_counts[s].get("clean", 0)   for s in splits]
    anomaly_counts = [flagged_counts[s].get("anomaly", 0) for s in splits]

    x = np.arange(len(splits))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x, clean_counts,   width, label="Clean",    color="#2ecc71")
    bars2 = ax.bar(x, anomaly_counts, width, bottom=clean_counts,
                   label="Anomaly / Flagged", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=12)
    ax.set_ylabel("Record Count", fontsize=12)
    ax.set_title("Adversarial Detection — Clean vs Flagged Records per Split",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)

    for bar, val in zip(bars2, anomaly_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "anomaly_detection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def plot_ml_comparison(ml_results: dict, output_dir: str):
    """
    Grouped bar chart comparing clean vs attack vs defense accuracy & F1.
    This is the key result chart for your report.
    """
    if not ml_results or "clean_baseline" not in ml_results:
        return

    scenarios = {
        "Clean\nBaseline":   ml_results.get("clean_baseline", {}),
        "Under\nAttack":     ml_results.get("adversarial_attack", {}),
        "After\nDefense":    ml_results.get("adversarial_defense", {}),
    }

    metrics_to_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    labels = list(scenarios.keys())
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, metric in enumerate(metrics_to_plot):
        values = [scenarios[s].get(metric, 0) for s in labels]
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=metric.replace("_", " ").title(), color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("MLlib Classification: Clean vs Adversarial vs Defended",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ml_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def plot_pipeline_flow(output_dir: str):
    """
    Simple diagram of the 7-stage pipeline using matplotlib patches.
    Great to include in your report as a pipeline architecture figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("1\nIngestion",    "#3498db"),
        ("2\nStorage",      "#2980b9"),
        ("3\nETL",          "#27ae60"),
        ("4\nDetection",    "#e67e22"),
        ("5\nAdaptation",   "#e74c3c"),
        ("6\nML",           "#9b59b6"),
        ("7\nAudit",        "#2c3e50"),
    ]

    box_w, box_h = 1.6, 1.4
    gap = 0.2
    y_center = 2.0

    for i, (label, color) in enumerate(stages):
        x = 0.3 + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_center, label,
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

        # Arrow to next stage
        if i < len(stages) - 1:
            ax_x = x + box_w + 0.02
            ax.annotate("", xy=(ax_x + gap - 0.02, y_center),
                        xytext=(ax_x, y_center),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    ax.set_title("A3L Big Data Pipeline — 7-Stage Architecture",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "pipeline_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved: {path}")


# ─────────────────────────────────────────────
#  STAGE 7 MAIN FUNCTION
# ─────────────────────────────────────────────

def stage_audit(
    pipeline_stats: dict,
    ml_results: dict,
    flagged_counts: dict,
    output_dir: str,
    thresholds: dict = None,
    elapsed_total: float = 0.0,
) -> str:
    """
    Stage 7 — Monitoring & Audit Layer
    Saves a structured audit log and generates all report figures.
    """
    banner("STAGE 7 — MONITORING & AUDIT")

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Build audit log
    logger = PipelineAuditLogger(output_dir)

    logger.record_stage("ingestion", {
        "splits_loaded": list(pipeline_stats.keys()),
        "total_records": sum(
            v.get("total_records", 0) for v in pipeline_stats.values()
        ),
    })

    logger.record_stage("etl", {
        "per_split_records": {
            k: v.get("total_records", 0) for k, v in pipeline_stats.items()
        },
        "avg_word_count_train": pipeline_stats.get("train", {}).get("word_count_mean", 0),
    })

    logger.record_stage("detection", {
        "thresholds": thresholds or {},
        "per_split_anomalies": flagged_counts,
    })

    robustness = ml_results.get("robustness_summary", {})
    logger.record_stage("ml_analytics", {
        "clean_accuracy":         robustness.get("clean_accuracy", "N/A"),
        "accuracy_under_attack":  robustness.get("accuracy_under_attack", "N/A"),
        "accuracy_after_defense": robustness.get("accuracy_after_defense", "N/A"),
        "accuracy_drop_pct":      robustness.get("accuracy_drop_pct", "N/A"),
        "accuracy_recovery_pct":  robustness.get("accuracy_recovery_pct", "N/A"),
    })

    logger.record_stage("pipeline_runtime", {
        "total_elapsed_seconds": round(elapsed_total, 2),
    })

    audit_path = logger.save()
    logger.print_summary()

    # Generate figures
    banner("GENERATING REPORT FIGURES")
    plot_pipeline_flow(figures_dir)
    plot_label_distribution(pipeline_stats, figures_dir)
    plot_anomaly_summary(flagged_counts, figures_dir)
    plot_ml_comparison(ml_results, figures_dir)

    print(f"\n  ✅  All figures saved in: {figures_dir}")
    print(f"  ✅  Audit log           : {audit_path}")
    return audit_path
