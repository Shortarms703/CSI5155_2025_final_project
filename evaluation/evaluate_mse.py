#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


def collect_image_paths(base_dir: Path) -> dict:
    """
    Recursively collects PNG image paths under base_dir, keyed by filename.
    This ignores the polygon_* directory structure so that misclassified
    predictions (saved under wrong polygon_* folder) can still be matched
    to the correct ground-truth by filename.
    """
    files = {}
    for root, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                full_path = Path(root) / fn
                files[fn] = full_path
    return files


def load_image_as_array(path: Path) -> np.ndarray:
    """
    Loads an image and returns a float32 numpy array scaled to [0, 1].
    Currently converts to grayscale; change to .convert("RGB") if needed.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_mse_per_image(gt_path: Path, pred_path: Path) -> float:
    gt = load_image_as_array(gt_path)
    pred = load_image_as_array(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_path} vs {pred_path}: {gt.shape} vs {pred.shape}")

    mse = float(np.mean((gt - pred) ** 2))
    return mse


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
        description="Compute MSE between ground-truth and 3 models "
                    "(baseline, with_classifier, without_classifier). "
                    "Outputs overall and per-polygon-type results."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Path to the 'output' folder that contains baseline/, ground_truth/, predictions_*/ etc.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="64x64",
        help="Resolution subfolder name (default: 64x64).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split subfolder name (default: test).",
    )
    parser.add_argument(
        "--eval_dirname",
        type=str,
        default="evaluation_results",
        help="Subfolder name under output_root to store results (default: evaluation_results).",
    )

    args = parser.parse_args()
    output_root = Path(args.output_root)
    resolution = args.resolution
    split = args.split
    eval_dir = output_root / args.eval_dirname
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Base raster directories
    gt_raster_dir = output_root / "ground_truth" / resolution / split / "raster"

    model_raster_dirs = {
        "baseline": output_root / "baseline" / resolution / split / "raster",
        "with_classifier": output_root / "predictions_with_classifier" / resolution / split / "raster",
        "without_classifier": output_root / "predictions_without_classifier" / resolution / split / "raster",
    }

    # Collect all ground-truth image paths by filename
    gt_files = collect_image_paths(gt_raster_dir)

    # For each model, collect prediction image paths by filename
    model_files = {
        model_name: collect_image_paths(path)
        for model_name, path in model_raster_dirs.items()
    }

    # Compute MSE per image for each model
    per_image_results = []
    overall_summary = {}

    # Also accumulate per-polygon-type stats in a structure:
    # by_polygon_type[model_name][polygon_type] = [mse_values...]
    by_polygon_type_values = defaultdict(lambda: defaultdict(list))

    for model_name, pred_files in model_files.items():
        mses = []
        missing_preds = 0
        compared_count = 0

        for filename, gt_path in gt_files.items():
            if filename not in pred_files:
                missing_preds += 1
                continue

            pred_path = pred_files[filename]

            # Polygon type from GT directory: .../raster/polygon_k/filename.png
            polygon_type = gt_path.parent.name  # e.g. "polygon_3"

            try:
                mse = compute_mse_per_image(gt_path, pred_path)
            except Exception as e:
                print(f"[{model_name}] Error computing MSE for {filename}: {e}")
                continue

            mses.append(mse)
            compared_count += 1
            by_polygon_type_values[model_name][polygon_type].append(mse)

            per_image_results.append({
                "model": model_name,
                "filename": filename,
                "polygon_type": polygon_type,
                "ground_truth_path": str(gt_path),
                "prediction_path": str(pred_path),
                "mse": mse,
            })

        if mses:
            mses_arr = np.array(mses)
            overall_summary[model_name] = {
                "num_samples_compared": int(compared_count),
                "num_missing_predictions": int(missing_preds),
                **{
                    f"{k}_mse": v
                    for k, v in compute_summary_stats(mses_arr).items()
                },
            }
        else:
            overall_summary[model_name] = {
                "num_samples_compared": 0,
                "num_missing_predictions": int(missing_preds),
                "mean_mse": None,
                "median_mse": None,
                "std_mse": None,
                "min_mse": None,
                "max_mse": None,
            }

    # Build per-polygon-type summary (overall + per polygon_type) for JSON
    by_polygon_type_summary = {}
    for model_name, poly_dict in by_polygon_type_values.items():
        by_polygon_type_summary[model_name] = {}
        for polygon_type, values in poly_dict.items():
            vals = np.array(values)
            by_polygon_type_summary[model_name][polygon_type] = {
                "num_samples": int(len(vals)),
                **{
                    f"{k}_mse": v
                    for k, v in compute_summary_stats(vals).items()
                },
            }

    # Save JSON results
    mse_json = {
        "description": "MSE between ground_truth and 3 models "
                       "(overall and per polygon_type).",
        "overall": overall_summary,
        "by_polygon_type": by_polygon_type_summary,
        "per_image": per_image_results,
    }

    mse_json_path = eval_dir / "mse_results.json"
    with open(mse_json_path, "w") as f:
        json.dump(mse_json, f, indent=2)
    print(f"Saved JSON results to: {mse_json_path}")

    # Create a DataFrame for CSV/LaTeX and plots
    if per_image_results:
        df = pd.DataFrame(per_image_results)

        # Save per-image table
        per_image_csv = eval_dir / "mse_per_image.csv"
        df.to_csv(per_image_csv, index=False)
        print(f"Saved per-image MSE table to: {per_image_csv}")

        # Summary table (overall + per polygon_type)
        summary_rows = []

        for model_name in model_files.keys():
            df_model = df[df["model"] == model_name]
            if df_model.empty:
                continue

            # Overall
            summary_rows.append({
                "model": model_name,
                "polygon_type": "all",
                "mean_mse": df_model["mse"].mean(),
                "median_mse": df_model["mse"].median(),
                "std_mse": df_model["mse"].std(),
                "min_mse": df_model["mse"].min(),
                "max_mse": df_model["mse"].max(),
                "num_samples": len(df_model),
            })

            # Per polygon_type
            for poly_type, df_group in df_model.groupby("polygon_type"):
                summary_rows.append({
                    "model": model_name,
                    "polygon_type": poly_type,
                    "mean_mse": df_group["mse"].mean(),
                    "median_mse": df_group["mse"].median(),
                    "std_mse": df_group["mse"].std(),
                    "min_mse": df_group["mse"].min(),
                    "max_mse": df_group["mse"].max(),
                    "num_samples": len(df_group),
                })

        df_summary = pd.DataFrame(summary_rows)

        summary_csv = eval_dir / "mse_summary_table.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Saved summary table (CSV) to: {summary_csv}")

        summary_tex = eval_dir / "mse_summary_table.tex"
        with open(summary_tex, "w") as f:
            f.write(df_summary.to_latex(index=False, float_format="%.6f"))
        print(f"Saved summary table (LaTeX) to: {summary_tex}")

        # Figures
        # 1. Boxplot of MSE distributions per model (overall)
        plt.figure(figsize=(8, 6))
        data_to_plot = [df[df["model"] == m]["mse"].values for m in model_files.keys()]
        labels = list(model_files.keys())
        plt.boxplot(data_to_plot, labels=labels, showfliers=False)
        plt.ylabel("MSE (pixel-wise)")
        plt.title("MSE distribution per model (overall)")
        plt.grid(axis="y", alpha=0.3)
        boxplot_path = eval_dir / "mse_boxplot_per_model.png"
        plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved MSE boxplot to: {boxplot_path}")

        # 2. Bar chart: mean MSE per polygon_type per model
        df_poly = df_summary[df_summary["polygon_type"] != "all"]
        if not df_poly.empty:
            pivot = df_poly.pivot(index="polygon_type", columns="model", values="mean_mse")
            pivot.plot(kind="bar", figsize=(8, 6))
            plt.ylabel("Mean MSE")
            plt.title("Mean MSE per polygon_type and model")
            plt.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=0)
            barplot_path = eval_dir / "mse_mean_per_polygon_type.png"
            plt.savefig(barplot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved MSE mean barplot to: {barplot_path}")


if __name__ == "__main__":
    main()
