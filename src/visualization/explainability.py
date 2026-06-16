"""
SHAP-based explainability for AFL Player Performance Predictor.

Production model is a single XGBRegressor predicting `Goals` for every
position. The `position` argument below is used only to label outputs
and group fairness comparisons — it does not route to a different model
or target variable.
Works with both LassoCV (LinearExplainer) and XGBoost (TreeExplainer).

Usage:
    from src.visualization.explainability import (
        get_explainer, explain_prediction, explain_batch,
        plot_waterfall, plot_force, plot_global_importance,
        plot_fairness_comparison, generate_explanation_dict
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import shap
from pathlib import Path
from typing import Union

# ── Constants ─────────────────────────────────────────────────────────────────

# Production model always predicts this single target, regardless of position.
MODEL_TARGET = "Goals"

# Audit groups from fairness_audit_framework.md
AGE_SEGMENTS = {
    "Young (<23)":   lambda df: df["Age"] < 23,
    "Prime (23-28)": lambda df: (df["Age"] >= 23) & (df["Age"] <= 28),
    "Veteran (>28)": lambda df: df["Age"] > 28,
}

OUTPUT_DIR = Path("reports/figures/explainability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Core: explainer factory ────────────────────────────────────────────────────

def get_explainer(model, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Return the appropriate SHAP explainer based on model type.
    - XGBoost / LightGBM / tree models → TreeExplainer (fast, exact)
    - Linear models (LassoCV, Ridge)   → LinearExplainer
    - Fallback                         → KernelExplainer (slow, model-agnostic)
    """
    model_type = type(model).__name__.lower()

    if any(t in model_type for t in ["xgb", "xgboost", "lgbm", "lightgbm",
                                      "randomforest", "gradientboosting",
                                      "decisiontree"]):
        return shap.TreeExplainer(model)

    if any(t in model_type for t in ["lasso", "ridge", "linear", "logistic",
                                      "elasticnet", "pipeline"]):
        return shap.LinearExplainer(model, X_background)

    # Fallback — slow but works on any sklearn-compatible model
    return shap.KernelExplainer(model.predict, shap.sample(X_background, 50))


# ── Single prediction explanation ─────────────────────────────────────────────

def explain_prediction(
    model,
    X_instance: pd.DataFrame,
    X_background: pd.DataFrame,
) -> dict:
    """
    Compute SHAP values for a single player prediction.

    Returns a dict with:
        shap_values    : np.ndarray of shape (n_features,)
        expected_value : float (model baseline)
        feature_names  : list[str]
        feature_values : list[float]
        top_features   : list[dict] sorted by |SHAP| descending
    """
    explainer = get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_instance)

    # Some explainers return a list (multi-output) — take first
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values).flatten()
    feature_names = list(X_instance.columns)
    feature_values = X_instance.values.flatten().tolist()

    expected_value = (
        explainer.expected_value[0]
        if hasattr(explainer.expected_value, "__len__")
        else float(explainer.expected_value)
    )

    top_features = sorted(
        [
            {
                "feature": name,
                "value": float(val),
                "shap_value": float(sv),
                "direction": "positive" if sv > 0 else "negative",
            }
            for name, val, sv in zip(feature_names, feature_values, shap_values)
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "top_features": top_features,
    }


def explain_batch(
    model,
    X_df: pd.DataFrame,
    X_background: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values for a batch of predictions. Returns (n_samples, n_features)."""
    explainer = get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.array(shap_values)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_waterfall(
    explanation: dict,
    position: str,
    player_id: str = "player",
    top_n: int = 10,
    save: bool = True,
) -> str:
    """
    Waterfall chart for a single prediction.
    Shows how each feature pushes the prediction above/below the baseline.
    Returns the file path of the saved figure.
    """
    top = explanation["top_features"][:top_n]
    features = [f["feature"] for f in top]
    shap_vals = [f["shap_value"] for f in top]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features[::-1], shap_vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(
        f"Prediction Explanation — {position} | {player_id}\n"
        f"Baseline: {explanation['expected_value']:.2f}  |  "
        f"Target: {MODEL_TARGET}"
    )

    # Add value labels
    for bar, val in zip(bars, shap_vals[::-1]):
        ax.text(
            val + (0.02 if val >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    path = OUTPUT_DIR / f"waterfall_{position.lower()}_{player_id}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_force(
    explanation: dict,
    position: str,
    player_id: str = "player",
    save: bool = True,
) -> str:
    """
    Force plot showing the push/pull of features on a single prediction.
    Returns the file path of the saved figure.
    """
    shap_vals = explanation["shap_values"]
    feature_names = explanation["feature_names"]
    expected_value = explanation["expected_value"]

    # Use SHAP's built-in force plot (saved as PNG via matplotlib)
    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=expected_value,
        data=np.array(explanation["feature_values"]),
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(14, 3))
    shap.plots.waterfall(shap_exp, max_display=15, show=False)
    path = OUTPUT_DIR / f"force_{position.lower()}_{player_id}.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_global_importance(
    model,
    X_df: pd.DataFrame,
    X_background: pd.DataFrame,
    position: str,
    top_n: int = 15,
    save: bool = True,
) -> str:
    """
    Global feature importance — mean |SHAP| across all predictions.
    Shows which features matter most overall for a given position.
    Returns the file path of the saved figure.
    """
    shap_values = explain_batch(model, X_df, X_background)
    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = list(X_df.columns)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#2ecc71")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(
        f"Global Feature Importance — {position} Model\n"
        f"Target: {MODEL_TARGET}  |  "
        f"n={len(X_df):,} predictions"
    )
    plt.tight_layout()

    path = OUTPUT_DIR / f"global_importance_{position.lower()}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_fairness_comparison(
    model,
    X_df: pd.DataFrame,
    X_background: pd.DataFrame,
    group_col: str,
    position: str,
    top_n: int = 10,
    save: bool = True,
) -> str:
    """
    Compare mean |SHAP| values across demographic/positional groups.
    Used in the fairness audit to check if different groups rely on different features.

    group_col: column in X_df to split on (e.g. 'PrimaryPosition', 'AgeSegment', 'Team')
    """
    groups = X_df[group_col].unique()
    feature_names = [c for c in X_df.columns if c != group_col]

    group_importances = {}
    for group in groups:
        group_df = X_df[X_df[group_col] == group][feature_names]
        if len(group_df) < 5:
            continue
        sv = explain_batch(model, group_df, X_background[feature_names])
        group_importances[str(group)] = np.abs(sv).mean(axis=0)

    if not group_importances:
        return ""

    # Top features by overall importance
    overall = np.mean(list(group_importances.values()), axis=0)
    top_idx = np.argsort(overall)[-top_n:]
    top_features = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top_features))
    width = 0.8 / len(group_importances)

    for i, (group, importances) in enumerate(group_importances.items()):
        vals = [importances[feature_names.index(f)] for f in top_features]
        ax.bar(x + i * width, vals, width, label=str(group), alpha=0.85)

    ax.set_xticks(x + width * (len(group_importances) - 1) / 2)
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title(
        f"Feature Importance by {group_col} — {position} Model\n"
        f"Fairness check: similar bars = equitable feature reliance"
    )
    ax.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / f"fairness_{position.lower()}_{group_col.lower()}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_age_segment_comparison(
    model,
    X_df: pd.DataFrame,
    X_background: pd.DataFrame,
    position: str,
    save: bool = True,
) -> str:
    """
    Fairness check across age segments (Young/Prime/Veteran).
    Key finding from Course 1: age strongly moderates treatment effects.
    """
    if "Age" not in X_df.columns:
        raise ValueError("X_df must contain an 'Age' column for age segment analysis")

    X_with_seg = X_df.copy()
    X_with_seg["AgeSegment"] = "Unknown"
    for seg, mask_fn in AGE_SEGMENTS.items():
        X_with_seg.loc[mask_fn(X_with_seg), "AgeSegment"] = seg

    return plot_fairness_comparison(
        model, X_with_seg, X_background,
        group_col="AgeSegment", position=position, save=save
    )


# ── API-friendly output ────────────────────────────────────────────────────────

def generate_explanation_dict(
    model,
    X_instance: pd.DataFrame,
    X_background: pd.DataFrame,
    position: str,
    player_id: str,
    top_n: int = 10,
) -> dict:
    """
    Returns a JSON-serializable explanation for the API endpoint
    GET /explain/{player_id}.

    Example response:
    {
        "player_id": "12345",
        "position": "Forward",
        "target": "Goals",
        "baseline": 3.21,
        "prediction": 5.84,
        "top_features": [
            {"feature": "MarksInside50", "value": 4.0, "shap_value": 2.1, "direction": "positive"},
            ...
        ]
    }
    """
    exp = explain_prediction(model, X_instance, X_background)
    prediction = exp["expected_value"] + sum(exp["shap_values"])

    return {
        "player_id": player_id,
        "position": position,
        "target": MODEL_TARGET,
        "baseline": round(exp["expected_value"], 4),
        "prediction": round(float(prediction), 4),
        "top_features": [
            {
                "feature": f["feature"],
                "value": round(f["value"], 4),
                "shap_value": round(f["shap_value"], 4),
                "direction": f["direction"],
            }
            for f in exp["top_features"][:top_n]
        ],
    }
