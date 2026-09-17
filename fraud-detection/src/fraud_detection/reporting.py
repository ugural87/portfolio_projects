from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


COLORS = {"navy": "#15324A", "blue": "#2878B5", "teal": "#2A9D8F", "amber": "#E9A23B", "red": "#D1495B", "gray": "#6B7280"}


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold", "axes.edgecolor": "#D1D5DB", "grid.alpha": 0.25})


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def eda_figures(frame, figure_dir: str | Path):
    setup_style()
    figure_dir = Path(figure_dir)
    counts = frame["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(["Legitimate", "Fraud"], counts.values, color=[COLORS["blue"], COLORS["red"]])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Extreme class imbalance")
    for i, value in enumerate(counts.values):
        ax.text(i, value * 1.12, f"{value:,}", ha="center", fontsize=9)
    save(fig, figure_dir / "01_class_imbalance.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, name, color in [(0, "Legitimate", COLORS["blue"]), (1, "Fraud", COLORS["red"])]:
        axes[0].hist(np.log1p(frame.loc[frame.Class.eq(label), "Amount"]), bins=55, density=True, alpha=0.55, label=name, color=color)
    axes[0].set_title("Transaction amount by class")
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].legend()
    temp = frame.assign(hour=(frame["Time"] / 3600).astype(int))
    hourly = temp.groupby("hour").agg(transactions=("Class", "size"), fraud_rate=("Class", "mean"))
    axes[1].plot(hourly.index, hourly.fraud_rate, marker="o", color=COLORS["teal"])
    axes[1].set_title("Fraud rate through elapsed hours")
    axes[1].set_xlabel("Elapsed hour")
    axes[1].set_ylabel("Fraud rate")
    save(fig, figure_dir / "02_amount_and_time.png")


def model_figures(artifact_dir: str | Path, figure_dir: str | Path):
    setup_style()
    artifact_dir, figure_dir = Path(artifact_dir), Path(figure_dir)
    valid = pd.read_csv(artifact_dir / "model_validation_comparison.csv").sort_values("validation_pr_auc")
    deep_path = artifact_dir / "deep_validation_comparison.csv"
    if deep_path.exists():
        deep = pd.read_csv(deep_path).rename(columns={"model": "candidate"})[["candidate", "validation_pr_auc"]]
        deep["candidate"] = deep["candidate"].str.replace("mlp_", "deep_mlp__", regex=False)
        valid = pd.concat([valid, deep], ignore_index=True).sort_values("validation_pr_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = valid.candidate.str.replace("__", " / ", regex=False).str.replace("_", " ", regex=False)
    colors = [COLORS["teal"] if name.startswith("deep_mlp") else COLORS["blue"] for name in valid.candidate]
    ax.barh(labels, valid.validation_pr_auc, color=colors)
    ax.set_xlabel("Validation PR-AUC")
    ax.set_title("Model selection on the chronological validation block")
    ax.set_xlim(0, max(valid.validation_pr_auc) * 1.12)
    for y, value in enumerate(valid.validation_pr_auc):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=8)
    save(fig, figure_dir / "03_model_validation.png")

    history_path = artifact_dir / "deep_training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name, group in history.groupby("loss"):
            ax.plot(group.epoch, group.validation_pr_auc, marker="o", label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation PR-AUC")
        ax.set_title("Deep MLP early-stopping trajectories")
        ax.legend()
        save(fig, figure_dir / "04_deep_training.png")


def decision_figures(artifact_dir: str | Path, figure_dir: str | Path):
    setup_style()
    artifact_dir, figure_dir = Path(artifact_dir), Path(figure_dir)
    pred = np.load(artifact_dir / "final_test_predictions.npz")
    y, p = pred["y"], pred["probability"]
    precision, recall, _ = precision_recall_curve(y, p)
    prevalence = y.mean()
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(recall, precision, color=COLORS["blue"], linewidth=2)
    axes[0].axhline(prevalence, color=COLORS["gray"], linestyle="--", label=f"Prevalence {prevalence:.3%}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Final holdout precision-recall")
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"])
    axes[1].plot(mean_pred, frac_pos, marker="o", color=COLORS["teal"])
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed fraud rate")
    axes[1].set_title("Final holdout calibration")
    save(fig, figure_dir / "05_final_pr_calibration.png")

    results = pd.read_csv(artifact_dir / "final_test_policy_results.csv")
    selected = results.loc[results.selected_on_policy_block].iloc[0]
    components = pd.Series({
        "Approve-all loss": selected.approve_all_cost,
        "Gross prevented": -selected.gross_prevented_loss,
        "Review cost": selected.review_cost_total,
        "Friction cost": selected.friction_cost_total,
        "Final cost": selected.realized_cost,
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [COLORS["red"], COLORS["teal"], COLORS["amber"], COLORS["amber"], COLORS["navy"]]
    ax.bar(components.index, components.values, color=colors)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Cost / benefit (currency units)")
    ax.set_title(f"Selected policy economics: {selected.policy}")
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(components.values):
        ax.text(i, value + (0.02 * max(abs(components.values))), f"{value:,.0f}", ha="center", fontsize=8)
    save(fig, figure_dir / "06_business_waterfall.png")

    sensitivity = pd.read_csv(artifact_dir / "policy_sensitivity.csv")
    pivot = sensitivity.pivot(index="capture_rate", columns="friction_cost", values="net_savings")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".0f", ax=ax)
    ax.set_title("Policy-block net savings sensitivity")
    ax.set_xlabel("Legitimate-customer friction cost")
    ax.set_ylabel("Review capture rate")
    save(fig, figure_dir / "07_sensitivity_heatmap.png")


def causal_figures(artifact_dir: str | Path, figure_dir: str | Path):
    setup_style()
    artifact_dir, figure_dir = Path(artifact_dir), Path(figure_dir)
    data = np.load(artifact_dir / "causal_plot_data.npz")
    ps, treatment = data["propensity"], data["treatment"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(ps[treatment == 0], bins=45, density=True, alpha=0.55, label="Untreated", color=COLORS["blue"])
    ax.hist(ps[treatment == 1], bins=45, density=True, alpha=0.55, label="Treated", color=COLORS["amber"])
    ax.set_xlabel("Cross-fitted propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Semi-synthetic treatment overlap")
    ax.legend()
    save(fig, figure_dir / "08_causal_overlap.png")

    balance = pd.read_csv(artifact_dir / "causal_balance.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(balance.smd_before, balance.feature, label="Before", color=COLORS["red"])
    ax.scatter(balance.smd_after_iptw, balance.feature, label="After IPTW", color=COLORS["teal"])
    ax.scatter(balance.smd_after_matching, balance.feature, label="After matching", color=COLORS["blue"])
    ax.axvline(-0.1, linestyle="--", color=COLORS["gray"])
    ax.axvline(0.1, linestyle="--", color=COLORS["gray"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance diagnostics")
    ax.legend()
    save(fig, figure_dir / "09_causal_balance.png")


def dashboard(artifact_dir: str | Path, figure_dir: str | Path):
    setup_style()
    artifact_dir, figure_dir = Path(artifact_dir), Path(figure_dir)
    business = json.loads((artifact_dir / "business_case.json").read_text())
    results = pd.read_csv(artifact_dir / "final_test_policy_results.csv")
    selected = results.loc[results.selected_on_policy_block].iloc[0]
    valid = pd.read_csv(artifact_dir / "model_validation_comparison.csv").sort_values("validation_pr_auc", ascending=False)
    deep = pd.read_csv(artifact_dir / "deep_validation_comparison.csv").rename(columns={"model": "candidate"})[["candidate", "validation_pr_auc"]]
    deep["candidate"] = deep["candidate"].str.replace("mlp_", "deep_mlp__", regex=False)
    valid = pd.concat([valid, deep], ignore_index=True).sort_values("validation_pr_auc", ascending=False)
    selected_model = json.loads((artifact_dir / "selected_model.json").read_text())["selected_candidate"]
    pred = np.load(artifact_dir / "final_test_predictions.npz")
    precision, recall, _ = precision_recall_curve(pred["y"], pred["probability"])

    fig = plt.figure(figsize=(14, 9), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 4, height_ratios=[0.75, 1.5, 1.5],
        left=0.08, right=0.98, bottom=0.08, top=0.86, hspace=0.62, wspace=0.38,
    )
    fig.suptitle("Credit Card Fraud Decision System", y=0.97, fontsize=18, fontweight="bold", color=COLORS["navy"])
    fig.text(0.62, 0.925, f"Predictive model: {selected_model} | Policy frozen before final holdout", ha="center", fontsize=10, color=COLORS["gray"])
    cards = [
        ("Selected policy", str(selected.policy)),
        ("Fraud recall", f"{selected.fraud_count_recall:.1%}"),
        ("Net savings", f"{selected.net_savings:,.0f} CU"),
        ("Alerts", f"{int(selected.alerts):,}"),
    ]
    for i, (label, value) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["navy"])
        ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=10, color=COLORS["gray"])
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, fill=False, edgecolor="#D1D5DB", linewidth=1, transform=ax.transAxes))

    ax1 = fig.add_subplot(gs[1, :2])
    top = valid.head(8).sort_values("validation_pr_auc")
    short_names = {
        "deep_mlp__focal": "Deep MLP / focal",
        "deep_mlp__weighted_bce": "Deep MLP / weighted BCE",
        "balanced_random_forest__balanced_bootstrap": "Balanced RF / bootstrap",
        "logistic__class_weight": "Logistic / class weight",
        "xgboost__scale_pos_weight": "XGBoost / class weight",
        "easy_ensemble__random_under_sampling": "EasyEnsemble / undersampling",
        "logistic__smote": "Logistic / SMOTE",
        "logistic__unweighted": "Logistic / unweighted",
        "lightgbm__scale_pos_weight": "LightGBM / class weight",
    }
    top_labels = top.candidate.map(short_names).fillna(top.candidate)
    top_colors = [COLORS["teal"] if name == selected_model else COLORS["blue"] for name in top.candidate]
    ax1.barh(top_labels, top.validation_pr_auc, color=top_colors)
    ax1.set_title("Validation PR-AUC")
    ax1.set_xlabel("Average precision")

    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(recall, precision, color=COLORS["teal"], linewidth=2)
    ax2.axhline(pred["y"].mean(), linestyle="--", color=COLORS["gray"])
    ax2.set_title("Final holdout precision-recall")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")

    ax3 = fig.add_subplot(gs[2, :2])
    plot_results = results.sort_values("net_savings")
    ax3.barh(plot_results.policy, plot_results.net_savings, color=[COLORS["teal"] if x else COLORS["blue"] for x in plot_results.selected_on_policy_block])
    ax3.axvline(0, color="#111827", linewidth=0.8)
    ax3.set_title("Final holdout net savings (frozen policy highlighted)")
    ax3.set_xlabel("Currency units")

    ax4 = fig.add_subplot(gs[2, 2:])
    interval = business["net_savings_95pct_bootstrap_interval"]
    values = [selected.gross_prevented_loss, selected.review_cost_total, selected.friction_cost_total, selected.net_savings]
    labels = ["Prevented loss", "Review cost", "Friction", "Net savings"]
    ax4.bar(labels, values, color=[COLORS["teal"], COLORS["amber"], COLORS["amber"], COLORS["navy"]])
    ax4.set_title(f"Selected economics\nSavings 95% CI: {interval['low']:,.0f} to {interval['high']:,.0f}")
    ax4.tick_params(axis="x", rotation=18)
    ax4.set_ylabel("Currency units")
    save(fig, figure_dir / "10_executive_dashboard.png")


def build_results_markdown(project_root: str | Path):
    root = Path(project_root)
    artifact_dir = root / "artifacts"
    audit = json.loads((artifact_dir / "data_audit.json").read_text())
    model = json.loads((artifact_dir / "selected_model.json").read_text())
    predictive = json.loads((artifact_dir / "selected_model_test_metrics.json").read_text())
    business = json.loads((artifact_dir / "business_case.json").read_text())
    deep = json.loads((artifact_dir / "deep_challenger.json").read_text())
    causal = json.loads((artifact_dir / "causal_diagnostics.json").read_text())
    test = business["test_metrics"]
    interval = business["net_savings_95pct_bootstrap_interval"]
    text = f"""# Executed Results

All figures and numbers below were produced by `scripts/run_pipeline.py` from the canonical ULB/OpenML raw dataset.

## Data

- Raw transactions: {audit['raw_rows']:,}
- Modeling transactions after exact-row deduplication: {audit['modeling_rows']:,}
- Exact duplicates removed: {audit['exact_duplicates_removed']:,}
- Modeling frauds: {audit['frauds_modeling']:,} ({audit['fraud_rate_modeling']:.4%})

## Selected predictive model

- Candidate: `{model['selected_candidate']}`
- Selection metric: chronological validation PR-AUC
- Strategy: `{model['strategy']}`
- Final holdout PR-AUC: {predictive['pr_auc']:.4f}
- Final holdout PR-AUC 95% interval: [{predictive['pr_auc_95pct_interval'][0]:.4f}, {predictive['pr_auc_95pct_interval'][2]:.4f}]

## Deep-learning challenger

- Selected loss: `{deep['selected_loss']}`
- Validation PR-AUC: {deep['validation_pr_auc']:.4f}
- Selected as overall predictive winner: {deep['selected_overall']}

## Frozen decision policy

- Policy selected before final test: `{business['selected_policy']}`
- Final test transactions: {int(test['transactions']):,}
- Alerts: {int(test['alerts']):,}
- Precision: {test['precision']:.2%}
- Fraud count recall: {test['fraud_count_recall']:.2%}
- Fraud amount capture: {test['fraud_amount_capture']:.2%}
- Gross prevented loss: {test['gross_prevented_loss']:,.2f} CU
- Net savings: {test['net_savings']:,.2f} CU
- Bootstrap 95% interval: [{interval['low']:,.2f}, {interval['high']:,.2f}] CU
- Illustrative savings per million transactions: {business['illustrative_savings_per_million_transactions']:,.2f} CU

These are scenario estimates under the documented cost assumptions, not realized savings from an identified bank.

## Semi-synthetic causal audit

- True constructed ATE: {causal['true_ate']:.4f} CU
- Cross-fitted AIPW ATE: {causal['aipw_ate']:.4f} CU
- IPTW effective sample size: {causal['iptw_effective_sample_size']:,.0f}
- Full-refit bootstrap repetitions: {causal['aipw_full_refit_bootstrap_95pct_ci']['repetitions']}

The causal section validates estimators against constructed potential outcomes. It does not claim that the public data identify a real review effect.
"""
    (root / "RESULTS.md").write_text(text, encoding="utf-8")
