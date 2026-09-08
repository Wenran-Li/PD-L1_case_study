import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. INPUT
# ============================================================

RF_CSV = "./outputs/rfdiffusion_bindcraft_full_v3/rfdiffusion_bindcraft_results.csv"
BC_CSV = "./outputs/bindcraft_pdl1/mpnn_design_stats.csv"

OUTDIR = "filter_attrition_results"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# 2. FILTER DEFINITIONS
#
# IMPORTANT:
# Replace these thresholds with EXACTLY the thresholds used in
# your BindCraft default_filters.json / RFdiffusion common filter.
# ============================================================

FILTERS = {
    "Complex pLDDT": {
        "column": "plddt",
        "direction": "min",
        "threshold": 0.80,
    },

    "Complex pTM": {
        "column": "ptm",
        "direction": "min",
        "threshold": 0.70,
    },

    "Interface pTM": {
        "column": "i_ptm",
        "direction": "min",
        "threshold": 0.65,
    },

    "Interface PAE": {
        "column": "i_pae",
        "direction": "max",
        "threshold": 0.35,
    },

    "Shape complementarity": {
        "column": "shape_complementarity",
        "direction": "min",
        "threshold": 0.55,
    },

    "Surface hydrophobicity": {
        "column": "surface_hydrophobicity",
        "direction": "max",
        "threshold": 0.35,
    },

    "Interface H-bonds": {
        "column": "interface_hbonds",
        "direction": "min",
        "threshold": 3,
    },

    "Unsatisfied interface H-bonds": {
        "column": "interface_unsat_hbonds",
        "direction": "max",
        "threshold": 5,
    },

    "Binder pLDDT": {
        "column": "binder_plddt",
        "direction": "min",
        "threshold": 0.80,
    },

    "Binder RMSD": {
        "column": "binder_rmsd",
        "direction": "max",
        "threshold": 2.0,
    },
}


# ============================================================
# 3. COLUMN ALIASES
#
# Add your actual CSV column names here if they differ.
# ============================================================

ALIASES = {

    "plddt": [
        "plddt",
        "pLDDT",
        "Average_pLDDT",
        "Complex_pLDDT"
    ],

    "ptm": [
        "ptm",
        "pTM",
        "Average_pTM",
        "Complex_pTM"
    ],

    "i_ptm": [
        "i_ptm",
        "i_pTM",
        "ipTM",
        "Interface_pTM",
        "Average_i_pTM"
    ],

    "i_pae": [
        "i_pae",
        "i_pAE",
        "ipAE",
        "Interface_pAE",
        "Average_i_pAE"
    ],

    "shape_complementarity": [
        "shape_complementarity",
        "ShapeComplementarity",
        "SC",
        "Interface_SC",
        "Average_ShapeComplementarity"
    ],

    "surface_hydrophobicity": [
        "surface_hydrophobicity",
        "SurfaceHydrophobicity",
        "hydrophobicity",
        "Average_Surface_Hydrophobicity"
    ],

    "interface_hbonds": [
        "interface_hbonds",
        "Interface_Hbonds",
        "interface_hbond",
        "HBonds",
        "Average_n_InterfaceHbonds"
    ],

    "interface_unsat_hbonds": [
        "interface_unsat_hbonds",
        "Interface_UnsatHbonds",
        "Unsat_Hbonds",
        "unsatisfied_hbonds",
        "Average_n_InterfaceUnsatHbonds"
    ],

    "binder_plddt": [
        "binder_plddt",
        "Binder_pLDDT",
        "binder_pLDDT",
        "Average_Binder_pLDDT"
    ],

    "binder_rmsd": [
        "binder_rmsd",
        "Binder_RMSD",
        "binder_RMSD",
        "Average_Binder_RMSD"
    ],
}


# ============================================================
# 4. COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    rename_dict = {}

    for standard_name, aliases in ALIASES.items():

        for alias in aliases:

            if alias in df.columns:
                rename_dict[alias] = standard_name
                break

    return df.rename(columns=rename_dict)


# ============================================================
# 5. PASS / FAIL FUNCTION
# ============================================================

def passes_filter(series, direction, threshold):

    if direction == "min":
        return series >= threshold

    elif direction == "max":
        return series <= threshold

    else:
        raise ValueError(
            f"Unknown filter direction: {direction}"
        )


# ============================================================
# 6. CHECK COLUMNS
# ============================================================

def check_columns(df, pipeline):

    required = [
        cfg["column"]
        for cfg in FILTERS.values()
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        print(f"\n[{pipeline}] Missing columns:")

        for x in missing:
            print("   ", x)

        print("\nAvailable columns:")
        print(list(df.columns))

        raise ValueError(
            f"{pipeline}: missing required columns."
        )


# ============================================================
# 7. CUMULATIVE SURVIVAL
# ============================================================

def cumulative_survival(df):

    current = pd.Series(
        True,
        index=df.index
    )

    results = []

    N0 = len(df)

    results.append({
        "Stage": "Full scoring",
        "N": N0,
        "Retention": 1.0,
    })

    for filter_name, cfg in FILTERS.items():

        passed = passes_filter(
            df[cfg["column"]],
            cfg["direction"],
            cfg["threshold"],
        )

        current &= passed.fillna(False)

        N = int(current.sum())

        results.append({
            "Stage": filter_name,
            "N": N,
            "Retention": N / N0,
        })

    return pd.DataFrame(results)


# ============================================================
# 8. INDEPENDENT FAILURE RATE
# ============================================================

def independent_failure(df):

    results = []

    for filter_name, cfg in FILTERS.items():

        valid = df[cfg["column"]].notna()

        passed = passes_filter(
            df.loc[valid, cfg["column"]],
            cfg["direction"],
            cfg["threshold"],
        )

        n_valid = valid.sum()
        n_fail = (~passed).sum()

        results.append({
            "Filter": filter_name,
            "N_valid": n_valid,
            "N_fail": n_fail,
            "Failure_rate": n_fail / n_valid
            if n_valid > 0 else np.nan
        })

    return pd.DataFrame(results)


# ============================================================
# 9. FIRST-FAILURE ANALYSIS
# ============================================================

def first_failure(df):

    first_fail = []

    for idx, row in df.iterrows():

        failure = "Passed all"

        for filter_name, cfg in FILTERS.items():

            value = row[cfg["column"]]

            if pd.isna(value):

                failure = f"Missing: {filter_name}"
                break

            if cfg["direction"] == "min":

                if value < cfg["threshold"]:
                    failure = filter_name
                    break

            elif cfg["direction"] == "max":

                if value > cfg["threshold"]:
                    failure = filter_name
                    break

        first_fail.append(failure)

    result = (
        pd.Series(first_fail)
        .value_counts()
        .rename_axis("First_failure")
        .reset_index(name="N")
    )

    result["Fraction"] = (
        result["N"] / len(df)
    )

    return result


# ============================================================
# 10. THRESHOLD STRINGENCY
#
# factor > 1 means STRICTER
#
# For minimum threshold:
#    threshold increases
#
# For maximum threshold:
#    threshold decreases
# ============================================================

def retention_under_stringency(
    df,
    factors=np.linspace(0.8, 1.2, 41)
):

    output = []

    for factor in factors:

        survive = pd.Series(
            True,
            index=df.index
        )

        for filter_name, cfg in FILTERS.items():

            threshold = cfg["threshold"]

            if cfg["direction"] == "min":

                adjusted_threshold = (
                    threshold * factor
                )

                passed = (
                    df[cfg["column"]]
                    >= adjusted_threshold
                )

            else:

                adjusted_threshold = (
                    threshold / factor
                )

                passed = (
                    df[cfg["column"]]
                    <= adjusted_threshold
                )

            survive &= passed.fillna(False)

        output.append({
            "Stringency": factor,
            "N_pass": survive.sum(),
            "Retention": survive.mean()
        })

    return pd.DataFrame(output)


# ============================================================
# 11. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_final_retention(
    df,
    n_boot=5000,
    seed=42
):

    rng = np.random.default_rng(seed)

    final_pass = pd.Series(
        True,
        index=df.index
    )

    for filter_name, cfg in FILTERS.items():

        passed = passes_filter(
            df[cfg["column"]],
            cfg["direction"],
            cfg["threshold"]
        )

        final_pass &= passed.fillna(False)

    x = final_pass.astype(int).values

    boots = []

    for _ in range(n_boot):

        sample = rng.choice(
            x,
            size=len(x),
            replace=True
        )

        boots.append(sample.mean())

    return {
        "retention": x.mean(),
        "CI_low": np.percentile(boots, 2.5),
        "CI_high": np.percentile(boots, 97.5),
    }


# ============================================================
# 12. LOAD DATA
# ============================================================

rf = pd.read_csv(RF_CSV)
bc = pd.read_csv(BC_CSV)

rf = normalize_columns(rf)
bc = normalize_columns(bc)

check_columns(rf, "RFdiffusion")
check_columns(bc, "BindCraft")


print("\nRFdiffusion full scoring:", len(rf))
print("BindCraft full scoring:", len(bc))


# ============================================================
# 13. RUN ANALYSIS
# ============================================================

rf_survival = cumulative_survival(rf)
bc_survival = cumulative_survival(bc)

rf_fail = independent_failure(rf)
bc_fail = independent_failure(bc)

rf_first = first_failure(rf)
bc_first = first_failure(bc)

rf_sens = retention_under_stringency(rf)
bc_sens = retention_under_stringency(bc)

rf_boot = bootstrap_final_retention(rf)
bc_boot = bootstrap_final_retention(bc)


# ============================================================
# 14. SAVE TABLES
# ============================================================

rf_survival.to_csv(
    f"{OUTDIR}/RFdiffusion_survival.csv",
    index=False
)

bc_survival.to_csv(
    f"{OUTDIR}/BindCraft_survival.csv",
    index=False
)

rf_fail.to_csv(
    f"{OUTDIR}/RFdiffusion_failure.csv",
    index=False
)

bc_fail.to_csv(
    f"{OUTDIR}/BindCraft_failure.csv",
    index=False
)

rf_first.to_csv(
    f"{OUTDIR}/RFdiffusion_first_failure.csv",
    index=False
)

bc_first.to_csv(
    f"{OUTDIR}/BindCraft_first_failure.csv",
    index=False
)

rf_sens.to_csv(
    f"{OUTDIR}/RFdiffusion_threshold_sensitivity.csv",
    index=False
)

bc_sens.to_csv(
    f"{OUTDIR}/BindCraft_threshold_sensitivity.csv",
    index=False
)


# ============================================================
# 15. FIGURE A:
# CUMULATIVE SURVIVAL
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(rf_survival))

ax.plot(
    x,
    rf_survival["Retention"] * 100,
    marker="o",
    linewidth=2,
    label="RFdiffusion"
)

ax.plot(
    x,
    bc_survival["Retention"] * 100,
    marker="o",
    linewidth=2,
    label="BindCraft"
)

ax.set_xticks(x)

ax.set_xticklabels(
    rf_survival["Stage"],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Candidates retained (%)")
ax.set_xlabel("Sequential filtering stage")

ax.set_ylim(0, 105)

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{OUTDIR}/panel_A_cumulative_survival.pdf",
    bbox_inches="tight"
)

fig.savefig(
    f"{OUTDIR}/panel_A_cumulative_survival.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 16. FIGURE B:
# INDEPENDENT FAILURE PROFILE
# ============================================================

merged_fail = pd.merge(
    rf_fail[
        ["Filter", "Failure_rate"]
    ],
    bc_fail[
        ["Filter", "Failure_rate"]
    ],
    on="Filter",
    suffixes=("_RFdiffusion", "_BindCraft")
)

x = np.arange(len(merged_fail))

width = 0.38

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    merged_fail[
        "Failure_rate_RFdiffusion"
    ] * 100,
    width,
    label="RFdiffusion"
)

ax.bar(
    x + width / 2,
    merged_fail[
        "Failure_rate_BindCraft"
    ] * 100,
    width,
    label="BindCraft"
)

ax.set_xticks(x)

ax.set_xticklabels(
    merged_fail["Filter"],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Candidates failing criterion (%)")
ax.set_xlabel("Independent filter")

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{OUTDIR}/panel_B_failure_profile.pdf",
    bbox_inches="tight"
)

fig.savefig(
    f"{OUTDIR}/panel_B_failure_profile.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 17. FIGURE C:
# THRESHOLD SENSITIVITY
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    rf_sens["Stringency"],
    rf_sens["Retention"] * 100,
    linewidth=2,
    label="RFdiffusion"
)

ax.plot(
    bc_sens["Stringency"],
    bc_sens["Retention"] * 100,
    linewidth=2,
    label="BindCraft"
)

ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Relative filter stringency"
)

ax.set_ylabel(
    "Final retention rate (%)"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{OUTDIR}/panel_C_threshold_sensitivity.pdf",
    bbox_inches="tight"
)

fig.savefig(
    f"{OUTDIR}/panel_C_threshold_sensitivity.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n============================")
print("FINAL RETENTION")
print("============================")

print(
    "RFdiffusion:",
    f"{100 * rf_boot['retention']:.2f}%",
    f"[95% CI: "
    f"{100 * rf_boot['CI_low']:.2f}–"
    f"{100 * rf_boot['CI_high']:.2f}%]"
)

print(
    "BindCraft:",
    f"{100 * bc_boot['retention']:.2f}%",
    f"[95% CI: "
    f"{100 * bc_boot['CI_low']:.2f}–"
    f"{100 * bc_boot['CI_high']:.2f}%]"
)

print("\nResults written to:")
print(OUTDIR)