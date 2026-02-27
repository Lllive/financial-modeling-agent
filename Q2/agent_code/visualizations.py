"""
可视化模块 — 生成所有图表
包含：数据概况、预处理效果、模型对比、ROC曲线、混淆矩阵、特征重要性
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# 全局样式
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
})
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ──────────────────────────────────────────────────────────
# 1. 数据概况可视化
# ──────────────────────────────────────────────────────────
def plot_data_overview(df):
    """数据概况：Y1分布、时间分布、特征缺失热图（前30列）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Data Overview — Financial Time Series", fontsize=14, fontweight="bold")

    # (A) Y1标签分布
    ax = axes[0]
    vc = df["Y1"].value_counts().sort_index()
    colors = ["#FF5722", "#9E9E9E", "#4CAF50"]
    bars = ax.bar(["-1 (Bearish)", "0 (Neutral)", "+1 (Bullish)"],
                  vc.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Y1 Label Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # (B) 每日样本数时间分布
    ax = axes[1]
    daily = df.groupby("trade_date").size()
    ax.fill_between(daily.index, daily.values, alpha=0.7, color="#2196F3")
    ax.set_title("Daily Sample Count (Time Distribution)", fontweight="bold")
    ax.set_ylabel("Samples per Day")
    ax.tick_params(axis="x", rotation=30)

    # (C) 特征缺失率（前50个特征）
    ax = axes[2]
    x_cols = [c for c in df.columns if c.startswith("X")][:50]
    miss_rates = df[x_cols].isnull().mean() * 100
    ax.barh(range(len(miss_rates)), miss_rates.values,
            color="#FF9800", alpha=0.8)
    ax.axvline(x=23, color="red", linestyle="--", linewidth=1, label="Mean")
    ax.set_title("Feature Missing Rate (X1-X50)", fontweight="bold")
    ax.set_xlabel("Missing Rate (%)")
    ax.set_yticks([])
    ax.legend()

    plt.tight_layout()
    plt.savefig("data_overview.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: data_overview.png")


# ──────────────────────────────────────────────────────────
# 2. 预处理效果可视化
# ──────────────────────────────────────────────────────────
def plot_preprocessing_effect(X_before, X_after, feature_indices=None):
    """对比标准化前后特征分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Preprocessing Effect — Standardization", fontsize=13, fontweight="bold")

    sample_cols = [0, 1, 2, 3, 4]

    ax = axes[0]
    for i, ci in enumerate(sample_cols):
        vals = X_before[:, ci]
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=50, alpha=0.5, label=f"X{ci+1}",
                color=PALETTE[i % len(PALETTE)])
    ax.set_title("Before Standardization", fontweight="bold")
    ax.set_xlabel("Value")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, ci in enumerate(sample_cols):
        ax.hist(X_after[:, ci], bins=50, alpha=0.5, label=f"X{ci+1}",
                color=PALETTE[i % len(PALETTE)])
    ax.set_title("After StandardScaler", fontweight="bold")
    ax.set_xlabel("Value (z-score)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("preprocessing_effect.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: preprocessing_effect.png")


# ──────────────────────────────────────────────────────────
# 3. 特征重要性（顶级特征）
# ──────────────────────────────────────────────────────────
def plot_feature_importance(state, top_n=30):
    """绘制特征重要性（互信息得分 + XGBoost内置重要性）"""
    mi_scores = state.get("feature_scores")
    x_cols = state.get("x_cols")
    models = state.get("models", {})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importance", fontsize=13, fontweight="bold")

    # (A) 互信息得分
    ax = axes[0]
    if mi_scores is not None:
        fi = state.get("feature_indices", np.argsort(mi_scores)[::-1][:top_n])
        top_idx = np.argsort(mi_scores)[::-1][:top_n]
        top_scores = mi_scores[top_idx]
        top_names = [x_cols[i] for i in top_idx]
        bars = ax.barh(range(top_n), top_scores[::-1], color="#2196F3", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_title(f"Mutual Information Score (Top {top_n})", fontweight="bold")
        ax.set_xlabel("MI Score")

    # (B) XGBoost特征重要性
    ax = axes[1]
    if "XGBoost" in models:
        xgb_imp = models["XGBoost"].feature_importances_
        sel_cols = state.get("selected_cols", x_cols)
        top_idx2 = np.argsort(xgb_imp)[::-1][:top_n]
        top_imp = xgb_imp[top_idx2]
        top_names2 = [sel_cols[i] for i in top_idx2]
        ax.barh(range(top_n), top_imp[::-1], color="#FF5722", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names2[::-1], fontsize=7)
        ax.set_title(f"XGBoost Feature Importance (Top {top_n})", fontweight="bold")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: feature_importance.png")


# ──────────────────────────────────────────────────────────
# 4. 所有模型ROC曲线对比
# ──────────────────────────────────────────────────────────
def plot_roc_curves(results, best_model_name):
    """绘制所有模型的ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")

    colors = {"LogisticRegression": "#9E9E9E",
              "XGBoost": "#FF5722",
              "LightGBM": "#4CAF50",
              "MLP_PyTorch": "#2196F3"}

    for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        fpr, tpr, _ = roc_curve(
            (res["pred"] * 0 + (res["proba"] >= 0).astype(int)),
            res["proba"]
        )
        # 使用真实标签计算ROC
        import numpy as _np
        y_test_arr = results[name].get("y_test")
        if y_test_arr is None:
            continue
        fpr, tpr, _ = roc_curve(y_test_arr, res["proba"])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == best_model_name else 1.5
        ls = "-" if name == best_model_name else "--"
        star = " ★" if name == best_model_name else ""
        c = colors.get(name, "#607D8B")
        ax.plot(fpr, tpr, color=c, linewidth=lw, linestyle=ls,
                label=f"{name}{star} (AUC={roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: roc_curves.png")


def plot_roc_curves_v2(results, y_test, best_model_name):
    """绘制所有模型的ROC曲线（需要传入y_test）"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")

    colors = {"LogisticRegression": "#9E9E9E",
              "XGBoost": "#FF5722",
              "LightGBM": "#4CAF50",
              "MLP_PyTorch": "#2196F3"}

    for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == best_model_name else 1.5
        ls = "-" if name == best_model_name else "--"
        star = " ★" if name == best_model_name else ""
        c = colors.get(name, "#607D8B")
        ax.plot(fpr, tpr, color=c, linewidth=lw, linestyle=ls,
                label=f"{name}{star} (AUC={roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: roc_curves.png")


# ──────────────────────────────────────────────────────────
# 5. 混淆矩阵（最优模型）
# ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bearish/Neutral (0)", "Bullish (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n"
                 f"(TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: confusion_matrix.png")


# ──────────────────────────────────────────────────────────
# 6. 模型性能对比柱状图
# ──────────────────────────────────────────────────────────
def plot_model_comparison(results):
    """横向对比4个指标: AUC, Precision, Recall, F1"""
    metrics = ["auc", "precision", "recall", "f1"]
    names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    colors2 = ["#9E9E9E", "#FF5722", "#4CAF50", "#2196F3"]
    for i, (name, c) in enumerate(zip(names, colors2)):
        vals = [results[name][m] for m in metrics]
        offset = (i - len(names)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=c, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(["AUC", "Precision", "Recall", "F1"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: model_comparison.png")


# ──────────────────────────────────────────────────────────
# 7. MLP训练曲线
# ──────────────────────────────────────────────────────────
def plot_mlp_training_curve(mlp_model):
    """绘制MLP训练Loss和验证AUC曲线"""
    losses = mlp_model.train_losses
    val_aucs = mlp_model.val_aucs

    if not losses:
        print("[WARN] No training history available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MLP Training Curves", fontsize=12, fontweight="bold")

    axes[0].plot(losses, color="#FF5722", linewidth=2)
    axes[0].set_title("Training Loss (BCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    if val_aucs:
        axes[1].plot(val_aucs, color="#2196F3", linewidth=2)
        axes[1].set_title("Validation AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].axhline(y=max(val_aucs), color="red", linestyle="--",
                        linewidth=1, label=f"Best AUC={max(val_aucs):.4f}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("mlp_training_curve.png", bbox_inches="tight", dpi=120)
    plt.show()
    print("✅ 图表已保存: mlp_training_curve.png")
