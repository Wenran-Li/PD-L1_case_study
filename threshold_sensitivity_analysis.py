#!/usr/bin/env python3
"""
Threshold-sensitivity analysis for the RFdiffusion–BindCraft PD-L1 case study.

This script does NOT rerun RFdiffusion, ProteinMPNN, AlphaFold2 or BindCraft.
It reads the already-generated result tables and varies one acceptance cutoff
at a time while keeping the remaining filters fixed at their default values.

Outputs
-------
1. Supplementary_Figure_threshold_sensitivity.png
2. Supplementary_Figure_threshold_sensitivity.pdf
3. threshold_sensitivity_values.csv

Example
-------
python threshold_sensitivity_analysis.py \
    --rfdiffusion_csv rfdiffusion_bindcraft_results.csv \
    --bindcraft_csv bindcraft_results.csv \
    --rf_total 2000 \
    --bc_total 1007 \
    --outdir supplementary_sensitivity
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------
# The first matching name found in each CSV is used.
COLUMN_ALIASES = {
    "pLDDT": [
        "Average_pLDDT", "pLDDT", "average_pLDDT", "average_plddt",
        "plddt", "avg_pLDDT", "avg_plddt"
    ],
    "pTM": [
        "Average_pTM", "pTM", "average_pTM", "average_ptm",
        "ptm", "avg_pTM", "avg_ptm"
    ],
    "i_pTM": [
        "Average_i_pTM", "i_pTM", "Interface_pTM", "interface_pTM",
        "iptm", "i_ptm"
    ],
    "i_pAE": [
        "Average_i_pAE", "i_pAE", "Interface_pAE", "interface_pAE",
        "ipae", "i_pae"
    ],
    "ShapeComplementarity": [
        "Average_ShapeComplementarity", "ShapeComplementarity",
        "shape_complementarity", "interface_sc", "Average_interface_sc"
    ],
    "n_InterfaceHbonds": [
        "Average_n_InterfaceHbonds", "n_InterfaceHbonds",
        "interface_interface_hbonds", "Interface_Hbonds",
        "interface_hbonds"
    ],
    "n_InterfaceUnsatHbonds": [
        "Average_n_InterfaceUnsatHbonds", "n_InterfaceUnsatHbonds",
        "interface_delta_unsat_hbonds", "Interface_Unsat_Hbonds",
        "interface_unsat_hbonds"
    ],
    "Surface_Hydrophobicity": [
        "Average_Surface_Hydrophobicity", "Surface_Hydrophobicity",
        "surface_hydrophobicity"
    ],
    "Binder_RMSD": [
        "Average_Binder_RMSD", "Binder_RMSD", "binder_rmsd"
    ],
}


# ---------------------------------------------------------------------
# Default acceptance criteria used in the manuscript
# ---------------------------------------------------------------------
# "higher" means value >= cutoff.
# "lower" means value <= cutoff.
DEFAULT_FILTERS = {
    "pLDDT": (0.80, "higher"),
    "pTM": (0.55, "higher"),
    "i_pTM": (0.50, "higher"),
    "i_pAE": (0.35, "lower"),
    "ShapeComplementarity": (0.60, "higher"),
    "n_InterfaceHbonds": (3.0, "higher"),
    "n_InterfaceUnsatHbonds": (4.0, "lower"),
    "Surface_Hydrophobicity": (0.35, "lower"),
    "Binder_RMSD": (3.5, "lower"),
}


# ---------------------------------------------------------------------
# Cutoff ranges for the six Supplementary Figure panels
# ---------------------------------------------------------------------
SENSITIVITY_RANGES = {
    "pLDDT": np.arange(0.70, 0.901, 0.025),
    "pTM": np.arange(0.45, 0.651, 0.025),
    "i_pTM": np.arange(0.40, 0.601, 0.025),
    "i_pAE": np.arange(0.25, 0.451, 0.025),
    "ShapeComplementarity": np.arange(0.50, 0.701, 0.025),
    "n_InterfaceHbonds": np.arange(1.0, 7.01, 1.0),
}

PANEL_LABELS = {
    "pLDDT": "pLDDT cutoff",
    "pTM": "pTM cutoff",
    "i_pTM": "Interface pTM cutoff",
    "i_pAE": "Interface PAE cutoff",
    "ShapeComplementarity": "Shape complementarity cutoff",
    "n_InterfaceHbonds": "Interface H-bonds cutoff",
}


def resolve_column(df, metric):
    """Return the real CSV column corresponding to a canonical metric name."""
    for candidate in COLUMN_ALIASES[metric]:
        if candidate in df.columns:
            return candidate
    return None


def prepare_dataframe(path, label):
    """Load a CSV and map its metric columns to canonical names."""
    df = pd.read_csv(path)

    mapped = pd.DataFrame(index=df.index)

    for metric in COLUMN_ALIASES:
        real_col = resolve_column(df, metric)
        if real_col is not None:
            mapped[metric] = pd.to_numeric(df[real_col], errors="coerce")
        else:
            mapped[metric] = np.nan

    # pLDDT is sometimes stored on a 0-100 scale.
    finite_plddt = mapped["pLDDT"].dropna()
    if len(finite_plddt) and finite_plddt.max() > 1.5:
        warnings.warn(
            f"{label}: pLDDT appears to be on a 0-100 scale; dividing by 100."
        )
        mapped["pLDDT"] = mapped["pLDDT"] / 100.0

    # Report mapping for reproducibility.
    print(f"\n[{label}] rows: {len(df)}")
    for metric in COLUMN_ALIASES:
        real_col = resolve_column(df, metric)
        if real_col is None:
            print(f"  {metric:28s} -> NOT FOUND")
        else:
            n = mapped[metric].notna().sum()
            print(f"  {metric:28s} -> {real_col}  (n={n})")

    return mapped


def candidate_pass_mask(df, varied_metric=None, varied_cutoff=None):
    """
    Apply the complete numeric filter set.

    If varied_metric is supplied, only that metric's cutoff is changed.
    Every other available criterion remains at its default cutoff.

    Missing metric values fail the corresponding filter. This means that
    candidates that did not reach complete downstream evaluation remain in
    the denominator but cannot be counted as retained at the final-filter stage.
    """
    mask = pd.Series(True, index=df.index, dtype=bool)

    for metric, (default_cutoff, direction) in DEFAULT_FILTERS.items():
        cutoff = (
            varied_cutoff
            if metric == varied_metric and varied_cutoff is not None
            else default_cutoff
        )

        values = df[metric]

        # If a whole column is unavailable in a dataset, do not silently
        # pretend that every candidate failed. Instead omit that criterion
        # and issue a warning once from validate_columns().
        if values.notna().sum() == 0:
            continue

        valid = values.notna()

        if direction == "higher":
            current = valid & (values >= cutoff)
        else:
            current = valid & (values <= cutoff)

        mask &= current

    return mask


def validate_columns(df, label):
    """
    Require all six sensitivity metrics.
    The three additional fixed filters are optional, but are used if present.
    """
    required = list(SENSITIVITY_RANGES.keys())
    missing_required = [m for m in required if df[m].notna().sum() == 0]

    if missing_required:
        raise ValueError(
            f"{label} is missing required metric columns: {missing_required}\n"
            "Check COLUMN_ALIASES near the top of this script and add the "
            "actual column names used by your CSV."
        )

    optional = [
        "n_InterfaceUnsatHbonds",
        "Surface_Hydrophobicity",
        "Binder_RMSD",
    ]
    missing_optional = [m for m in optional if df[m].notna().sum() == 0]

    if missing_optional:
        warnings.warn(
            f"{label}: optional fixed filters unavailable and therefore not "
            f"applied: {missing_optional}"
        )


def retention_percent(df, mask, denominator):
    """
    Retention relative to the requested workflow denominator.

    denominator can be larger than the number of rows in the CSV, which is useful
    if a BindCraft table contains only candidates that reached metric evaluation.
    """
    numerator = int(mask.sum())
    return 100.0 * numerator / float(denominator), numerator


def run_sensitivity(rf, bc, rf_total, bc_total):
    rows = []

    for metric, cutoffs in SENSITIVITY_RANGES.items():
        default_cutoff, direction = DEFAULT_FILTERS[metric]

        for cutoff in cutoffs:
            rf_mask = candidate_pass_mask(
                rf, varied_metric=metric, varied_cutoff=float(cutoff)
            )
            bc_mask = candidate_pass_mask(
                bc, varied_metric=metric, varied_cutoff=float(cutoff)
            )

            rf_pct, rf_n = retention_percent(rf, rf_mask, rf_total)
            bc_pct, bc_n = retention_percent(bc, bc_mask, bc_total)

            rows.append({
                "metric": metric,
                "direction": direction,
                "cutoff": float(cutoff),
                "default_cutoff": float(default_cutoff),
                "RFdiffusion_retained_n": rf_n,
                "RFdiffusion_denominator": int(rf_total),
                "RFdiffusion_retention_percent": rf_pct,
                "BindCraft_retained_n": bc_n,
                "BindCraft_denominator": int(bc_total),
                "BindCraft_retention_percent": bc_pct,
                "difference_BindCraft_minus_RF_pp": bc_pct - rf_pct,
            })

    return pd.DataFrame(rows)


def make_figure(results, output_png, output_pdf):
    """
    Create a six-panel Supplementary Figure.
    """
    metrics = list(SENSITIVITY_RANGES.keys())

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12.5, 7.4),
        constrained_layout=True
    )
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        d = results[results["metric"] == metric].sort_values("cutoff")

        ax.plot(
            d["cutoff"],
            d["RFdiffusion_retention_percent"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="RFdiffusion"
        )
        ax.plot(
            d["cutoff"],
            d["BindCraft_retention_percent"],
            marker="s",
            linewidth=1.8,
            markersize=4,
            label="BindCraft"
        )

        default_cutoff = DEFAULT_FILTERS[metric][0]
        ax.axvline(
            default_cutoff,
            linestyle="--",
            linewidth=1.2,
            label="Default cutoff" if i == 0 else None
        )

        ax.set_xlabel(PANEL_LABELS[metric])
        ax.set_ylabel("Retained candidates (%)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)

        panel_letter = chr(ord("A") + i)
        ax.text(
            0.02, 0.98,
            panel_letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=12
        )

    # One common legend for the whole figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)
    )

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def print_default_point_summary(results):
    print("\nRetention at the manuscript default cutoffs:")
    for metric in SENSITIVITY_RANGES:
        default = DEFAULT_FILTERS[metric][0]
        d = results[results["metric"] == metric].copy()

        # Avoid float-equality problems.
        idx = (d["cutoff"] - default).abs().idxmin()
        row = d.loc[idx]

        print(
            f"  {metric:24s} cutoff={row['cutoff']:.3f} | "
            f"RF {row['RFdiffusion_retained_n']}/"
            f"{row['RFdiffusion_denominator']} "
            f"({row['RFdiffusion_retention_percent']:.2f}%) | "
            f"BC {row['BindCraft_retained_n']}/"
            f"{row['BindCraft_denominator']} "
            f"({row['BindCraft_retention_percent']:.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Threshold-sensitivity analysis for RFdiffusion vs BindCraft."
    )

    parser.add_argument(
        "--rfdiffusion_csv",
        required=True,
        help="RFdiffusion downstream result CSV."
    )
    parser.add_argument(
        "--bindcraft_csv",
        required=True,
        help="BindCraft result CSV containing the corresponding metrics."
    )
    parser.add_argument(
        "--rf_total",
        type=int,
        default=None,
        help=(
            "Retention denominator for RFdiffusion. "
            "For sequence-level workflow retention use 2000. "
            "If omitted, number of rows in the CSV is used."
        )
    )
    parser.add_argument(
        "--bc_total",
        type=int,
        default=None,
        help=(
            "Retention denominator for BindCraft. "
            "For sequence-level workflow retention use 1007. "
            "If omitted, number of rows in the CSV is used."
        )
    )
    parser.add_argument(
        "--outdir",
        default="supplementary_sensitivity",
        help="Output directory."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rf = prepare_dataframe(args.rfdiffusion_csv, "RFdiffusion")
    bc = prepare_dataframe(args.bindcraft_csv, "BindCraft")

    validate_columns(rf, "RFdiffusion")
    validate_columns(bc, "BindCraft")

    rf_total = args.rf_total if args.rf_total is not None else len(rf)
    bc_total = args.bc_total if args.bc_total is not None else len(bc)

    if rf_total < len(rf):
        raise ValueError(
            f"--rf_total ({rf_total}) cannot be smaller than RF CSV rows ({len(rf)})."
        )
    if bc_total < len(bc):
        raise ValueError(
            f"--bc_total ({bc_total}) cannot be smaller than BindCraft CSV rows ({len(bc)})."
        )

    results = run_sensitivity(
        rf=rf,
        bc=bc,
        rf_total=rf_total,
        bc_total=bc_total,
    )

    csv_out = os.path.join(
        args.outdir,
        "threshold_sensitivity_values.csv"
    )
    png_out = os.path.join(
        args.outdir,
        "Supplementary_Figure_threshold_sensitivity.png"
    )
    pdf_out = os.path.join(
        args.outdir,
        "Supplementary_Figure_threshold_sensitivity.pdf"
    )

    results.to_csv(csv_out, index=False)
    make_figure(results, png_out, pdf_out)

    print_default_point_summary(results)

    print("\nSaved:")
    print(" ", csv_out)
    print(" ", png_out)
    print(" ", pdf_out)


if __name__ == "__main__":
    main()
