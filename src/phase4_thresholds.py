"""Phase 4: probability thresholds for operational decision zones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_utils import CLASS_ORDER, configure_plots, write_text


ZONE_CONFIDENCE = "Zona de Confianza"
ZONE_UNCERTAINTY = "Zona de Incertidumbre"
ZONE_REJECTION = "Zona de Rechazo"
ZONE_ORDER = [ZONE_CONFIDENCE, ZONE_UNCERTAINTY, ZONE_REJECTION]


def resolve_model_prefix(model: str, summary_path: Path) -> str:
    """Resolve the probability-column prefix to use."""
    if model in {"mlp", "ensemble"}:
        return model

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_model = summary["best_model_by_test_f1_weighted"]
    return "ensemble" if "Ensemble" in best_model else "mlp"


def assign_zone(probability: float, confidence: float, rejection: float) -> str:
    """Assign one operational zone from the maximum posterior probability."""
    if probability >= confidence:
        return ZONE_CONFIDENCE
    if probability >= rejection:
        return ZONE_UNCERTAINTY
    return ZONE_REJECTION


def apply_threshold_policy(
    predictions: pd.DataFrame,
    prefix: str,
    confidence: float,
    rejection: float,
) -> pd.DataFrame:
    """Apply confidence, uncertainty and rejection thresholds."""
    proba_cols = [f"{prefix}_proba_{cls}" for cls in CLASS_ORDER]
    missing = [col for col in proba_cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    proba = predictions[proba_cols].to_numpy()
    pred_idx = np.argmax(proba, axis=1)
    max_proba = np.max(proba, axis=1)

    out = predictions.copy()
    out["selected_model"] = prefix
    out["threshold_pred"] = [CLASS_ORDER[idx] for idx in pred_idx]
    out["max_probability"] = max_proba
    out["decision_zone"] = [
        assign_zone(p, confidence=confidence, rejection=rejection) for p in max_proba
    ]
    out["is_correct"] = out["threshold_pred"] == out["species_id"]
    out["requires_human_review"] = out["decision_zone"] != ZONE_CONFIDENCE
    return out


def summarize_zones(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create global and species-level summaries for the policy."""
    rows = []
    total = len(df)
    for zone in ZONE_ORDER:
        zone_df = df[df["decision_zone"] == zone]
        rows.append(
            {
                "decision_zone": zone,
                "n_observations": int(len(zone_df)),
                "coverage": len(zone_df) / total if total else 0.0,
                "accuracy": zone_df["is_correct"].mean() if len(zone_df) else np.nan,
                "mean_probability": zone_df["max_probability"].mean()
                if len(zone_df)
                else np.nan,
            }
        )

    species_summary = (
        df.groupby(["species_id", "decision_zone"], observed=True)
        .agg(
            n_observations=("recording_id", "count"),
            accuracy=("is_correct", "mean"),
            mean_probability=("max_probability", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), species_summary


def plot_threshold_summary(
    thresholded: pd.DataFrame,
    zone_summary: pd.DataFrame,
    confidence: float,
    rejection: float,
    output_path: Path,
) -> None:
    """Save operational threshold diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(
        zone_summary["decision_zone"],
        zone_summary["n_observations"],
        color=["#1D9E75", "#EF9F27", "#E24B4A"],
    )
    axes[0].set_title("Distribucion por zona", fontsize=16)
    axes[0].set_ylabel("Numero de observaciones", fontsize=14)
    axes[0].tick_params(axis="x", labelrotation=15, labelsize=14)
    axes[0].tick_params(axis="y", labelsize=14)

    axes[1].hist(thresholded["max_probability"], bins=18, color="#378ADD", alpha=0.85)
    axes[1].axvline(confidence, color="#1D9E75", linestyle="--", label="0.85")
    axes[1].axvline(rejection, color="#E24B4A", linestyle="--", label="0.40")
    axes[1].set_title("Probabilidad maxima posterior", fontsize=16)
    axes[1].set_xlabel("Probabilidad maxima", fontsize=14)
    axes[1].set_ylabel("Frecuencia", fontsize=14)
    axes[1].tick_params(labelsize=14)
    axes[1].legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_latex_float(value: float) -> str:
    """Format a nullable float for LaTeX."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.3f}"


def write_phase4_latex(
    output_path: Path,
    prefix: str,
    confidence: float,
    rejection: float,
    zone_summary: pd.DataFrame,
) -> None:
    """Write the Phase 4 LaTeX analysis."""
    rows = []
    for _, row in zone_summary.iterrows():
        rows.append(
            f"{row['decision_zone']} & {int(row['n_observations'])} & "
            f"{row['coverage']:.3f} & {format_latex_float(row['accuracy'])} & "
            f"{format_latex_float(row['mean_probability'])} \\\\"
        )

    model_name = "MLP TensorFlow" if prefix == "mlp" else "Soft Voting Ensemble"
    content = rf"""\subsection{{MLOps y l\'ogica de umbrales}}
Se propuso una pol\'itica operacional basada en la probabilidad posterior m\'axima del modelo {model_name}. Las predicciones con $P \geq {confidence:.2f}$ fueron asignadas a la zona de confianza, las predicciones con ${rejection:.2f} \leq P < {confidence:.2f}$ fueron asignadas a la zona de incertidumbre y las predicciones con $P < {rejection:.2f}$ fueron asignadas a la zona de rechazo.

\begin{{table}}[H]
\centering
\caption{{Resumen operativo de zonas de decisi\'on.}}
\begin{{tabular}}{{lrrrr}}
\hline
Zona & Observaciones & Cobertura & Accuracy & Prob. media \\
\hline
{chr(10).join(rows)}
\hline
\end{{tabular}}
\end{{table}}

La zona de confianza fue definida para automatizar decisiones de clasificaci\'on con alta certeza. La zona de incertidumbre fue destinada a revisi\'on humana o validaci\'on secundaria, mientras que la zona de rechazo fue reservada para casos donde no deber\'ia emitirse una clasificaci\'on autom\'atica. Esta l\'ogica permite controlar el riesgo operativo al separar cobertura del sistema y confiabilidad de las decisiones aceptadas.
"""
    write_text(output_path, content)


def run_phase4(args: argparse.Namespace) -> dict:
    """Execute the threshold policy workflow."""
    configure_plots(args.font_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = resolve_model_prefix(args.model, Path(args.phase3_summary))
    predictions = pd.read_csv(args.predictions)
    thresholded = apply_threshold_policy(
        predictions,
        prefix=prefix,
        confidence=args.confidence_threshold,
        rejection=args.rejection_threshold,
    )
    zone_summary, species_summary = summarize_zones(thresholded)

    thresholded.to_csv(output_dir / "phase4_thresholded_predictions.csv", index=False)
    zone_summary.to_csv(output_dir / "phase4_zone_summary.csv", index=False)
    species_summary.to_csv(output_dir / "phase4_species_zone_summary.csv", index=False)

    plot_threshold_summary(
        thresholded,
        zone_summary,
        args.confidence_threshold,
        args.rejection_threshold,
        output_dir / "phase4_threshold_diagnostics.png",
    )
    write_phase4_latex(
        Path(args.report_path),
        prefix,
        args.confidence_threshold,
        args.rejection_threshold,
        zone_summary,
    )

    summary = {
        "selected_model_prefix": prefix,
        "confidence_threshold": args.confidence_threshold,
        "rejection_threshold": args.rejection_threshold,
        "zone_summary": zone_summary.to_dict(orient="records"),
        "outputs": {
            "thresholded_predictions": str(
                output_dir / "phase4_thresholded_predictions.csv"
            ),
            "zone_summary": str(output_dir / "phase4_zone_summary.csv"),
            "species_summary": str(output_dir / "phase4_species_zone_summary.csv"),
            "diagnostics_plot": str(output_dir / "phase4_threshold_diagnostics.png"),
            "latex_report": args.report_path,
        },
    }
    write_text(output_dir / "phase4_summary.json", json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 4 threshold policy.")
    parser.add_argument("--predictions", default="outputs/phase3/phase3_test_predictions.csv")
    parser.add_argument("--phase3-summary", default="outputs/phase3/phase3_summary.json")
    parser.add_argument("--output-dir", default="outputs/phase4")
    parser.add_argument("--report-path", default="report/phase4_analysis.tex")
    parser.add_argument("--model", choices=["best", "mlp", "ensemble"], default="best")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--rejection-threshold", type=float, default=0.40)
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    summary = run_phase4(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
