#!/usr/bin/env python3

import os
import sys
import glob
import json
import csv
import shutil
import argparse
import traceback
import numpy as np
import pyrosetta as pr

# ------------------------------------------------------------
# BindCraft imports
# ------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import *

from colabdesign import mk_afdesign_model, clear_mem


from Bio.PDB import PDBParser, Superimposer
import numpy as np


def target_aligned_binder_rmsd(
    reference_pdb,
    mobile_pdb,
    target_chain="A",
    binder_chain="B"
):
    """
    Align mobile complex to reference using target-chain CA atoms,
    then calculate binder-chain CA RMSD without further alignment.
    """

    parser = PDBParser(QUIET=True)

    ref_structure = parser.get_structure(
        "reference",
        reference_pdb
    )

    mob_structure = parser.get_structure(
        "mobile",
        mobile_pdb
    )

    ref_target = ref_structure[0][target_chain]
    mob_target = mob_structure[0][target_chain]

    ref_target_res = [
        r for r in ref_target
        if r.id[0] == " " and "CA" in r
    ]

    mob_target_res = [
        r for r in mob_target
        if r.id[0] == " " and "CA" in r
    ]

    n_target = min(
        len(ref_target_res),
        len(mob_target_res)
    )

    ref_atoms = [
        ref_target_res[i]["CA"]
        for i in range(n_target)
    ]

    mob_atoms = [
        mob_target_res[i]["CA"]
        for i in range(n_target)
    ]

    # --------------------------------------------------
    # Align mobile TARGET onto reference TARGET
    # --------------------------------------------------

    sup = Superimposer()

    sup.set_atoms(
        ref_atoms,
        mob_atoms
    )

    # transform entire mobile structure,
    # including binder chain B
    sup.apply(
        list(mob_structure.get_atoms())
    )

    # --------------------------------------------------
    # Calculate binder CA RMSD after TARGET alignment
    # --------------------------------------------------

    ref_binder = ref_structure[0][binder_chain]
    mob_binder = mob_structure[0][binder_chain]

    ref_binder_res = [
        r for r in ref_binder
        if r.id[0] == " " and "CA" in r
    ]

    mob_binder_res = [
        r for r in mob_binder
        if r.id[0] == " " and "CA" in r
    ]

    n_binder = min(
        len(ref_binder_res),
        len(mob_binder_res)
    )

    if n_binder == 0:
        raise ValueError(
            "No matching binder CA atoms found."
        )

    ref_xyz = np.array([
        ref_binder_res[i]["CA"].coord
        for i in range(n_binder)
    ])

    mob_xyz = np.array([
        mob_binder_res[i]["CA"].coord
        for i in range(n_binder)
    ])

    diff = ref_xyz - mob_xyz

    rmsd = np.sqrt(
        np.mean(
            np.sum(diff ** 2, axis=1)
        )
    )

    return round(float(rmsd), 2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_binder_length(pdb_file, binder_chain="B"):
    """Count unique residues in binder chain."""
    residues = set()

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM") and line[21].strip() == binder_chain:
                residues.add((line[22:26].strip(), line[26].strip()))

    return len(residues)


def apply_filters(flat_data, filters):
    """
    Apply BindCraft JSON thresholds to a flattened result dictionary.

    Returns
    -------
    passed : bool
    failed : list[str]
    """
    failed = []

    for name, rule in filters.items():

        if name not in flat_data:
            continue

        # InterfaceAAs is nested and handled separately below
        if not isinstance(rule, dict):
            continue

        # Normal scalar filter
        if "threshold" in rule:

            threshold = rule["threshold"]

            if threshold is None:
                continue

            value = flat_data[name]

            if value is None:
                continue

            higher = rule.get("higher", True)

            try:
                value = float(value)
                threshold = float(threshold)
            except (ValueError, TypeError):
                continue

            if higher:
                if value < threshold:
                    failed.append(
                        f"{name}={value:.3f} < {threshold}"
                    )
            else:
                if value > threshold:
                    failed.append(
                        f"{name}={value:.3f} > {threshold}"
                    )

    return len(failed) == 0, failed


def flatten_model_statistics(complex_stats,
                             complex_averages,
                             binder_stats,
                             binder_averages):

    row = {}

    # -----------------------------
    # Complex AF2 + interface stats
    # -----------------------------
    for model_num, stats in complex_stats.items():
        for key, value in stats.items():
            if key == "InterfaceAAs":
                continue
            row[f"{model_num}_{key}"] = value

    for key, value in complex_averages.items():
        if key == "InterfaceAAs":
            continue
        row[f"Average_{key}"] = value

    # -----------------------------
    # Binder-alone AF2 stats
    # -----------------------------
    for model_num, stats in binder_stats.items():
        for key, value in stats.items():
            row[f"{model_num}_Binder_{key}"] = value

    for key, value in binder_averages.items():
        row[f"Average_Binder_{key}"] = value

    return row


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--target_settings",
        default="./settings_target/PDL1.json"
    )

    parser.add_argument(
        "--filters",
        default="./settings_filters/default_filters.json"
    )

    parser.add_argument(
        "--advanced",
        default="./settings_advanced/default_4stage_multimer.json"
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--binder_chain",
        default="B"
    )

    parser.add_argument(
        "--num_seqs",
        type=int,
        default=None,
        help="Override BindCraft number of ProteinMPNN sequences"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load BindCraft settings
    # ------------------------------------------------------------

    target_settings = load_json(args.target_settings)
    filters = load_json(args.filters)
    advanced_settings = load_json(args.advanced)

    if args.num_seqs is not None:
        advanced_settings["num_seqs"] = args.num_seqs

    # IMPORTANT:
    # RFdiffusion did not design a meaningful binder sequence.
    # Therefore do NOT freeze its interface residues during ProteinMPNN.
    advanced_settings["mpnn_fix_interface"] = False

    if not pr.rosetta.basic.was_init_called():
        print("Initialising PyRosetta...")
        dalphaball = os.path.abspath("./functions/DAlphaBall.gcc")

        if not os.path.exists(dalphaball):
            raise FileNotFoundError(
                f"DAlphaBall not found: {dalphaball}"
            )

        print(f"Using DAlphaBall: {dalphaball}")

        pr.init(
            options=(
                "-mute all "
                "-ignore_unrecognized_res true "
                "-ignore_zero_occupancy false "
                "-load_PDB_components false "
                "-relax:default_repeats 1 "
                f"-holes:dalphaball {dalphaball}"
            )
        )


    binder_chain = args.binder_chain

    # Same validation models as normal BindCraft
    prediction_models = advanced_settings.get(
        "validation_model_indices",
        [0, 1]
    )

    if prediction_models is None:
        prediction_models = [0, 1]

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------

    output_dir = os.path.abspath(args.output_dir)

    paths = {
        "MPNN": os.path.join(output_dir, "MPNN"),
        "MPNN/Relaxed": os.path.join(output_dir, "MPNN", "Relaxed"),
        "MPNN/Binder": os.path.join(output_dir, "MPNN", "Binder"),
        "Accepted": os.path.join(output_dir, "Accepted"),
        "Rejected": os.path.join(output_dir, "Rejected"),
        "Trajectory": os.path.join(output_dir, "Trajectory"),
        "Trajectory/Relaxed": os.path.join(
            output_dir, "Trajectory", "Relaxed"
        ),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    failure_csv = os.path.join(output_dir, "failures.csv")

    if not os.path.exists(failure_csv):
        with open(failure_csv, "w") as f:
            f.write("Filter,Count\n")

    # ------------------------------------------------------------
    # Input RFdiffusion structures
    # ------------------------------------------------------------

    pdb_files = sorted(
        glob.glob(
            os.path.join(
                os.path.abspath(args.input_dir),
                "PDL1_binder_*.pdb"
            )
        )
    )

    print("=" * 70)
    print("RFdiffusion → BindCraft validation")
    print("=" * 70)

    print(f"Input directory : {args.input_dir}")
    print(f"Structures      : {len(pdb_files)}")
    print(f"Target chain    : A")
    print(f"Binder chain    : {binder_chain}")
    print(f"AF2 models      : {prediction_models}")
    print(f"MPNN sequences  : {advanced_settings['num_seqs']}")
    print("=" * 70)

    if len(pdb_files) == 0:
        raise RuntimeError(
            f"No PDL1_binder_*.pdb found in {args.input_dir}"
        )

    all_rows = []

    # ============================================================
    # RFdiffusion structures
    # ============================================================

    for pdb_index, trajectory_pdb in enumerate(pdb_files):

        design_name = os.path.splitext(
            os.path.basename(trajectory_pdb)
        )[0]

        print("\n")
        print("#" * 70)
        print(
            f"[{pdb_index + 1}/{len(pdb_files)}] "
            f"{design_name}"
        )
        print("#" * 70)

        try:

            length = get_binder_length(
                trajectory_pdb,
                binder_chain
            )

            print(f"Binder length: {length}")

            # ----------------------------------------------------
            # Analyze RFdiffusion interface
            # ----------------------------------------------------

            trajectory_relaxed = os.path.join(
                paths["Trajectory/Relaxed"],
                design_name + ".pdb"
            )

            if not os.path.exists(trajectory_relaxed):
                print("Relaxing RFdiffusion structure...")
                pr_relax(
                    trajectory_pdb,
                    trajectory_relaxed
                )

            print("Calculating RFdiffusion interface...")

            (
                trajectory_interface_scores,
                trajectory_interface_AA,
                trajectory_interface_residues
            ) = score_interface(
                trajectory_relaxed,
                binder_chain
            )

            print(
                "Interface residues:",
                trajectory_interface_residues
            )

            # ----------------------------------------------------
            # ProteinMPNN
            # ----------------------------------------------------

            print("\nRunning ProteinMPNN...")

            mpnn_trajectories = mpnn_gen_sequence(
                trajectory_pdb,
                binder_chain,
                trajectory_interface_residues,
                advanced_settings
            )

            candidates = []

            for n in range(len(mpnn_trajectories["seq"])):

                sequence = mpnn_trajectories["seq"][n][-length:]

                candidates.append({
                    "seq": sequence,
                    "score": mpnn_trajectories["score"][n],
                    "seqid": mpnn_trajectories["seqid"][n]
                })

            # Remove duplicate sequences
            unique = {}

            for candidate in candidates:
                unique[candidate["seq"]] = candidate

            candidates = sorted(
                unique.values(),
                key=lambda x: x["score"]
            )

            print(
                f"ProteinMPNN produced "
                f"{len(candidates)} unique sequences."
            )

            # ----------------------------------------------------
            # Compile AF2 ONCE for this backbone
            # ----------------------------------------------------

            clear_mem()

            multimer_validation = advanced_settings.get(
                "use_multimer_validation",
                True
            )

            complex_prediction_model = mk_afdesign_model(
                protocol="binder",
                num_recycles=advanced_settings[
                    "num_recycles_validation"
                ],
                data_dir=advanced_settings["af_params_dir"],
                use_multimer=multimer_validation
            )

            # Use the SAME target preparation strategy as BindCraft
            complex_prediction_model.prep_inputs(
                pdb_filename=target_settings["starting_pdb"],
                chain=target_settings["chains"],
                binder_len=length,
                rm_target_seq=advanced_settings[
                    "rm_template_seq_predict"
                ],
                rm_target_sc=advanced_settings[
                    "rm_template_sc_predict"
                ]
            )

            binder_prediction_model = mk_afdesign_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=advanced_settings[
                    "num_recycles_validation"
                ],
                data_dir=advanced_settings["af_params_dir"],
                use_multimer=multimer_validation
            )

            binder_prediction_model.prep_inputs(
                length=length
            )

            # ====================================================
            # ProteinMPNN sequences
            # ====================================================

            for mpnn_n, candidate in enumerate(candidates):

                mpnn_design_name = (
                    f"{design_name}_mpnn{mpnn_n}"
                )

                sequence = candidate["seq"]

                print("\n" + "-" * 60)
                print(mpnn_design_name)
                print(
                    f"MPNN score: {candidate['score']:.3f}"
                )

                row = {
                    "Design": design_name,
                    "MPNN_design": mpnn_design_name,
                    "Binder_length": length,
                    "Sequence": sequence,
                    "MPNN_score": candidate["score"],
                    "MPNN_seqid": candidate["seqid"]
                }

                # -----------------------------------------------
                # AF2 complex
                # -----------------------------------------------

                print("AF2 complex prediction...")

                complex_stats, pass_af2 = predict_binder_complex(
                    complex_prediction_model,
                    sequence,
                    mpnn_design_name,
                    target_settings["starting_pdb"],
                    target_settings["chains"],
                    length,
                    trajectory_pdb,
                    prediction_models,
                    advanced_settings,
                    filters,
                    paths,
                    failure_csv
                )

                if not pass_af2:

                    print("FAIL: base AF2 filters")

                    row["Status"] = "FAIL_AF2"

                    # preserve whatever AF2 values exist
                    for model_num, stats in complex_stats.items():
                        for key, value in stats.items():
                            row[f"{model_num}_{key}"] = value

                    all_rows.append(row)
                    continue

                # -----------------------------------------------
                # PyRosetta / structural metrics
                # -----------------------------------------------

                for model_num in prediction_models:

                    model_id = model_num + 1

                    complex_pdb = os.path.join(
                        paths["MPNN"],
                        f"{mpnn_design_name}_model{model_id}.pdb"
                    )

                    relaxed_pdb = os.path.join(
                        paths["MPNN/Relaxed"],
                        f"{mpnn_design_name}_model{model_id}.pdb"
                    )

                    if not os.path.exists(complex_pdb):
                        continue

                    clashes = calculate_clash_score(
                        complex_pdb
                    )

                    relaxed_clashes = calculate_clash_score(
                        relaxed_pdb
                    )

                    (
                        interface_scores,
                        interface_AA,
                        interface_residues
                    ) = score_interface(
                        relaxed_pdb,
                        binder_chain
                    )

                    (
                        alpha,
                        beta,
                        loops,
                        alpha_interface,
                        beta_interface,
                        loops_interface,
                        i_plddt,
                        ss_plddt
                    ) = calc_ss_percentage(
                        complex_pdb,
                        advanced_settings,
                        binder_chain
                    )

                    hotspot_rmsd = target_aligned_binder_rmsd(
                        trajectory_pdb,
                        complex_pdb,
                        target_chain="A",
                        binder_chain=binder_chain
                    )

                    target_rmsd = target_pdb_rmsd(
                        complex_pdb,
                        target_settings["starting_pdb"],
                        target_settings["chains"]
                    )

                    complex_stats[model_id].update({

                        "i_pLDDT": i_plddt,
                        "ss_pLDDT": ss_plddt,

                        "Unrelaxed_Clashes": clashes,
                        "Relaxed_Clashes": relaxed_clashes,

                        "Binder_Energy_Score":
                            interface_scores["binder_score"],

                        "Surface_Hydrophobicity":
                            interface_scores[
                                "surface_hydrophobicity"
                            ],

                        "ShapeComplementarity":
                            interface_scores["interface_sc"],

                        "PackStat":
                            interface_scores[
                                "interface_packstat"
                            ],

                        "dG":
                            interface_scores["interface_dG"],

                        "dSASA":
                            interface_scores["interface_dSASA"],

                        "dG/dSASA":
                            interface_scores[
                                "interface_dG_SASA_ratio"
                            ],

                        "Interface_SASA_%":
                            interface_scores[
                                "interface_fraction"
                            ],

                        "Interface_Hydrophobicity":
                            interface_scores[
                                "interface_hydrophobicity"
                            ],

                        "n_InterfaceResidues":
                            interface_scores[
                                "interface_nres"
                            ],

                        "n_InterfaceHbonds":
                            interface_scores[
                                "interface_interface_hbonds"
                            ],

                        "InterfaceHbondsPercentage":
                            interface_scores[
                                "interface_hbond_percentage"
                            ],

                        "n_InterfaceUnsatHbonds":
                            interface_scores[
                                "interface_delta_unsat_hbonds"
                            ],

                        "InterfaceUnsatHbondsPercentage":
                            interface_scores[
                                "interface_delta_unsat_hbonds_percentage"
                            ],

                        "InterfaceAAs":
                            interface_AA,

                        "Interface_Helix%":
                            alpha_interface,

                        "Interface_BetaSheet%":
                            beta_interface,

                        "Interface_Loop%":
                            loops_interface,

                        "Binder_Helix%":
                            alpha,

                        "Binder_BetaSheet%":
                            beta,

                        "Binder_Loop%":
                            loops,

                        "Hotspot_RMSD":
                            hotspot_rmsd,

                        "Target_RMSD":
                            target_rmsd
                    })

                # -----------------------------------------------
                # Average complex statistics
                # -----------------------------------------------

                complex_averages = calculate_averages(
                    complex_stats,
                    handle_aa=True
                )

                # -----------------------------------------------
                # AF2 binder alone
                # -----------------------------------------------

                print("AF2 binder-alone prediction...")

                binder_stats = predict_binder_alone(
                    binder_prediction_model,
                    sequence,
                    mpnn_design_name,
                    length,
                    trajectory_pdb,
                    binder_chain,
                    prediction_models,
                    advanced_settings,
                    paths
                )

                # Calculate Binder RMSD ourselves
                for model_num in prediction_models:

                    model_id = model_num + 1

                    binder_pdb = os.path.join(
                        paths["MPNN/Binder"],
                        f"{mpnn_design_name}_model{model_id}.pdb"
                    )

                    if not os.path.exists(binder_pdb):
                        continue

                    binder_rmsd = unaligned_rmsd(
                        trajectory_pdb,
                        binder_pdb,
                        binder_chain,
                        "A"
                    )

                    binder_stats[model_id][
                        "RMSD"
                    ] = binder_rmsd

                binder_averages = calculate_averages(
                    binder_stats
                )

                # -----------------------------------------------
                # Flatten into BindCraft naming convention
                # -----------------------------------------------

                flat = flatten_model_statistics(
                    complex_stats,
                    complex_averages,
                    binder_stats,
                    binder_averages
                )

                row.update(flat)

                # -----------------------------------------------
                # Apply default_filters.json
                # -----------------------------------------------

                passed, failed = apply_filters(
                    flat,
                    filters
                )

                if passed:

                    row["Status"] = "PASS"
                    row["Failed_filters"] = ""

                    print("***** PASS *****")

                    # Copy best complex
                    source = os.path.join(
                        paths["MPNN/Relaxed"],
                        f"{mpnn_design_name}_model1.pdb"
                    )

                    if os.path.exists(source):
                        shutil.copy(
                            source,
                            os.path.join(
                                paths["Accepted"],
                                os.path.basename(source)
                            )
                        )

                else:

                    row["Status"] = "FAIL"
                    row["Failed_filters"] = "; ".join(
                        failed
                    )

                    print(
                        "FAIL:",
                        "; ".join(failed)
                    )

                all_rows.append(row)

        except Exception as e:

            print("\nERROR processing", design_name)
            print(str(e))
            traceback.print_exc()

            all_rows.append({
                "Design": design_name,
                "Status": "ERROR",
                "Error": str(e)
            })

        # --------------------------------------------------------
        # Save checkpoint after EVERY backbone
        # --------------------------------------------------------

        if all_rows:

            keys = sorted({
                key
                for r in all_rows
                for key in r.keys()
            })

            result_csv = os.path.join(
                output_dir,
                "rfdiffusion_bindcraft_results.csv"
            )

            with open(
                result_csv,
                "w",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=keys
                )

                writer.writeheader()
                writer.writerows(all_rows)

    print("\n")
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    pass_n = sum(
        r.get("Status") == "PASS"
        for r in all_rows
    )

    fail_n = sum(
        str(r.get("Status", "")).startswith("FAIL")
        for r in all_rows
    )

    print(f"PASS : {pass_n}")
    print(f"FAIL : {fail_n}")
    print(f"TOTAL: {len(all_rows)}")

    print(
        "\nResults:",
        os.path.join(
            output_dir,
            "rfdiffusion_bindcraft_results.csv"
        )
    )


if __name__ == "__main__":
    main()