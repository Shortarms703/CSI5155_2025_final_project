#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_json_as_dict(path: Path) -> Dict[str, Any]:
    """
    Loads a JSON list of objects and returns a dict keyed by 'filename'.
    Assumes each entry has a unique 'filename' key.
    """
    with open(path, "r") as f:
        data = json.load(f)

    mapping = {}
    for item in data:
        fname = item.get("filename")
        if fname is None:
            continue
        mapping[fname] = item
    return mapping


def extract_points(entry: Dict[str, Any]) -> np.ndarray:
    """
    Extracts an Nx2 array of points from a JSON entry.
    Prefer 'normalized_points' if present and non-empty; otherwise use 'points'.
    """
    if "normalized_points" in entry and entry["normalized_points"]:
        pts = np.array(entry["normalized_points"], dtype=np.float64)
    else:
        pts = np.array(entry["points"], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Invalid points shape: {pts.shape}")
    return pts


def hausdorff_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Computes the symmetric discrete Hausdorff distance between two point sets
    points_a (Na x 2) and points_b (Nb x 2), using Euclidean distance.

    H(A, B) = max( sup_a inf_b d(a, b), sup_b inf_a d(b, a) )
    """
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    d_ab = dist_matrix.min(axis=1).max()
    d_ba = dist_matrix.min(axis=0).max()
    return float(max(d_ab, d_ba))


def compute_summary_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Hausdorff distance between ground_truth polygons "
                    "and 3 models (baseline, with_classifier, without_classifier). "
                    "Outputs overall and per-polygon-type results."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Path to the 'output' folder that contains the JSON files.",
    )
    parser.add_argument(
        "--eval_dirname",
        type=str,
        default="evaluation_results",
        help="Subfolder name under output_root to store results (default: evaluation_results).",
    )

    args = parser.parse_args()
    output_root = Path(args.output_root)
    eval_dir = output_root / args.eval_dirname
    eval_dir.mkdir(parents=True, exist_ok=True)

    # JSON paths
    gt_json_path = output_root / "ground_truth_test_set.json"
    baseline_json_path = output_root / "baseline_test_set.json"
    with_clf_json_path = output_root / "predictions_with_classifier.json"
    without_clf_json_path = output_root / "predictions_without_classifier.json"

    # Load as dicts keyed by filename
    gt_dict = load_json_as_dict(gt_json_path)
    model_dicts = {
        "baseline": load_json_as_dict(baseline_json_path),
        "with_classifier": load_json_as_dict(with_clf_json_path),
        "without_classifier": load_json_as_dict(without_clf_json_path),
    }

    per_sample_results: List[Dict[str, Any]] = []
    overall_summary = {}
    # by_polygon_type[model_name][polygon_type_gt] = [hd_values...]
    by_polygon_type_values = {}

    for model_name, model_data in model_dicts.items():
        dists = []
        num_compared = 0
        num_missing = 0
        num_errors = 0
        by_polygon_type_values[model_name] = {}

        for filename, gt_entry in gt_dict.items():
            if filename not in model_data:
                num_missing += 1
                continue

            pred_entry = model_data[filename]
            try:
                pts_gt = extract_points(gt_entry)
                pts_pred = extract_points(pred_entry)
                hd = hausdorff_distance(pts_gt, pts_pred)
            except Exception as e:
                print(f"[{model_name}] Error computing Hausdorff for {filename}: {e}")
                num_errors += 1
                continue

            num_compared += 1
            dists.append(hd)

            polygon_type_gt = gt_entry.get("polygon_type")  # e.g. "polygon_3"
            if polygon_type_gt not in by_polygon_type_values[model_name]:
                by_polygon_type_values[model_name][polygon_type_gt] = []
            by_polygon_type_values[model_name][polygon_type_gt].append(hd)

            per_sample_results.append({
                "model": model_name,
                "filename": filename,
                "polygon_type_gt": polygon_type_gt,
                "polygon_type_pred": pred_entry.get("polygon_type"),
                "num_vertices_gt": gt_entry.get("num_vertices"),
                "num_vertices_pred": pred_entry.get("num_vertices"),
                "hausdorff_distance": hd,
            })

        if dists:
            dists_arr = np.array(dists)
            overall_summary[model_name] = {
                "num_samples_compared": int(num_compared),
                "num_missing_predictions": int(num_missing),
                "num_error_samples": int(num_errors),
                **{
                    f"{k}_hausdorff": v
                    for k, v in compute_summary_stats(dists_arr).items()
                },
            }
        else:
            overall_summary[model_name] = {
                "num_samples_compared": 0,
                "num_missing_predictions": int(num_missing),
                "num_error_samples": int(num_errors),
                "mean_hausdorff": None,
                "median_hausdorff": None,
                "std_hausdorff": None,
                "min_hausdorff": None,
                "max_hausdorff": None,
            }

    # Per-polygon-type summary
    by_polygon_type_summary = {}
    for model_name, poly_dict in by_polygon_type_values.items():
        by_polygon_type_summary[model_name] = {}
        for polygon_type, values in poly_dict.items():
            vals = np.array(values)
            by_polygon_type_summary[model_name][polygon_type] = {
                "num_samples": int(len(vals)),
                **{
                    f"{k}_hausdorff": v
                    for k, v in compute_summary_stats(vals).items()
                },
            }

    # Save JSON results
    hd_json = {
        "description": "Hausdorff distances between ground_truth and 3 models "
                       "(overall and per ground-truth polygon_type).",
        "overall": overall_summary,
        "by_polygon_type": by_polygon_type_summary,
        "per_sample": per_sample_results,
    }

    hd_json_path = eval_dir / "hausdorff_results.json"
    with open(hd_json_path, "w") as f:
        json.dump(hd_json, f, indent=2)
    print(f"Saved JSON results to: {hd_json_path}")

    # Create DataFrame and tables/figures
    if per_sample_results:
        df = pd.DataFrame(per_sample_results)

        # Save per-sample table
        per_sample_csv = eval_dir / "hausdorff_per_sample.csv"
        df.to_csv(per_sample_csv, index=False)
        print(f"Saved per-sample Hausdorff table to: {per_sample_csv}")

        # Summary table (overall + per polygon_type_gt)
        summary_rows = []
        for model_name in model_dicts.keys():
            df_model = df[df["model"] == model_name]
            if df_model.empty:
                continue

            # Overall
            summary_rows.append({
                "model": model_name,
                "polygon_type_gt": "all",
                "mean_hausdorff": df_model["hausdorff_distance"].mean(),
                "median_hausdorff": df_model["hausdorff_distance"].median(),
                "std_hausdorff": df_model["hausdorff_distance"].std(),
                "min_hausdorff": df_model["hausdorff_distance"].min(),
                "max_hausdorff": df_model["hausdorff_distance"].max(),
                "num_samples": len(df_model),
            })

            # Per GT polygon_type
            for poly_type, df_group in df_model.groupby("polygon_type_gt"):
                summary_rows.append({
                    "model": model_name,
                    "polygon_type_gt": poly_type,
                    "mean_hausdorff": df_group["hausdorff_distance"].mean(),
                    "median_hausdorff": df_group["hausdorff_distance"].median(),
                    "std_hausdorff": df_group["hausdorff_distance"].std(),
                    "min_hausdorff": df_group["hausdorff_distance"].min(),
                    "max_hausdorff": df_group["hausdorff_distance"].max(),
                    "num_samples": len(df_group),
                })

        df_summary = pd.DataFrame(summary_rows)

        summary_csv = eval_dir / "hausdorff_summary_table.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Saved summary table (CSV) to: {summary_csv}")

        summary_tex = eval_dir / "hausdorff_summary_table.tex"
        with open(summary_tex, "w") as f:
            f.write(df_summary.to_latex(index=False, float_format="%.6f"))
        print(f"Saved summary table (LaTeX) to: {summary_tex}")

        # Figures
        # 1. Boxplot of Hausdorff distances per model (overall)
        plt.figure(figsize=(8, 6))
        data_to_plot = [df[df["model"] == m]["hausdorff_distance"].values for m in model_dicts.keys()]
        labels = list(model_dicts.keys())
        plt.boxplot(data_to_plot, labels=labels, showfliers=False)
        plt.ylabel("Hausdorff distance (normalized units)")
        plt.title("Hausdorff distance distribution per model (overall)")
        plt.grid(axis="y", alpha=0.3)
        boxplot_path = eval_dir / "hausdorff_boxplot_per_model.png"
        plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Hausdorff boxplot to: {boxplot_path}")

        # 2. Bar chart: mean Hausdorff distance per GT polygon_type per model
        df_poly = df_summary[df_summary["polygon_type_gt"] != "all"]
        if not df_poly.empty:
            pivot = df_poly.pivot(index="polygon_type_gt", columns="model", values="mean_hausdorff")
            pivot.plot(kind="bar", figsize=(8, 6))
            plt.ylabel("Mean Hausdorff distance")
            plt.title("Mean Hausdorff distance per ground-truth polygon_type and model")
            plt.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=0)
            barplot_path = eval_dir / "hausdorff_mean_per_polygon_type.png"
            plt.savefig(barplot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved Hausdorff mean barplot to: {barplot_path}")


if __name__ == "__main__":
    main()
