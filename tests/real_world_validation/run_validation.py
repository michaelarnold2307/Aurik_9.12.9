#!/usr/bin/env python3
"""
Run Validation Suite on Original vs AURIK-Processed Files

Compares objective metrics between original test files and AURIK-processed results.

Usage:
    python run_validation.py
    python run_validation.py --category vinyl
    python run_validation.py --output validation_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from validation_suite import ValidationSuite


def _workspace_output_path(path_value: Path) -> Path:
    """Allow generated validation reports only below the working directory."""
    workspace = Path.cwd().resolve()
    output_path = path_value.resolve()
    try:
        output_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside the working directory: {path_value}") from exc
    return output_path


def run_validation(test_library: Path, processed_dir: Path, output_file: Path, category: str = None):  # type: ignore[assignment]
    """Run validation on all files and generate comparison report."""

    output_file = _workspace_output_path(output_file)

    # Initialize validation suite
    validator = ValidationSuite()

    # Get categories to validate
    if category:
        categories = [category]
    else:
        categories = ["vinyl", "tape", "digital", "vocals"]

    print(f"\n{'=' * 80}")
    print("VALIDATION SUITE - Objective Metrics Comparison")
    print("Comparing: Original vs AURIK-Processed")
    print(f"Categories: {', '.join(categories)}")
    print(f"{'=' * 80}\n")

    # Collect all results
    results: Any = {"summary": {}, "per_category": {}, "per_file": []}

    total_files = 0
    snr_improvements = []
    thd_changes = []

    for cat in categories:
        print(f"\n{'─' * 80}")
        print(f"Category: {cat.upper()}")
        print(f"{'─' * 80}\n")

        cat_results = []

        # Get original files
        original_dir = test_library / cat
        if not original_dir.exists():
            print(f"⚠️  Category directory not found: {original_dir}")
            continue

        original_files = sorted(original_dir.glob("*.wav"))

        for orig_file in original_files:
            # Find corresponding processed file
            processed_file = processed_dir / cat / f"{orig_file.stem}_restored.wav"

            if not processed_file.exists():
                print(f"⚠️  Processed file not found: {processed_file.name}")
                continue

            total_files += 1

            print(f"Analyzing: {orig_file.name}")

            # Analyze original
            orig_metrics = validator.analyze_file(orig_file)

            # Analyze processed
            proc_metrics = validator.analyze_file(processed_file, reference_path=orig_file)

            # Compute improvements
            snr_improvement = proc_metrics["snr"] - orig_metrics["snr"]
            thd_change = proc_metrics["thd"] - orig_metrics["thd"]

            snr_improvements.append(snr_improvement)
            thd_changes.append(thd_change)

            # Store results
            file_result = {
                "filename": orig_file.name,
                "category": cat,
                "original": {
                    "snr": orig_metrics["snr"],
                    "thd": orig_metrics["thd"],
                    "spectral_centroid": orig_metrics["spectral"]["centroid"],
                    "spectral_flatness": orig_metrics["spectral"]["flatness"],
                    "dynamic_range_db": orig_metrics["dynamics"]["dynamic_range_db"],
                },
                "processed": {
                    "snr": proc_metrics["snr"],
                    "thd": proc_metrics["thd"],
                    "spectral_centroid": proc_metrics["spectral"]["centroid"],
                    "spectral_flatness": proc_metrics["spectral"]["flatness"],
                    "dynamic_range_db": proc_metrics["dynamics"]["dynamic_range_db"],
                },
                "improvements": {"snr_improvement": snr_improvement, "thd_change": thd_change},
            }

            if "reference_based" in proc_metrics:
                file_result["reference_based"] = proc_metrics["reference_based"]

            results["per_file"].append(file_result)
            cat_results.append(file_result)

            # Print summary
            print(f"  SNR: {orig_metrics['snr']:.2f} → {proc_metrics['snr']:.2f} dB ({snr_improvement:+.2f} dB)")
            print(f"  THD: {orig_metrics['thd']:.3f}% → {proc_metrics['thd']:.3f}% ({thd_change:+.3f}%)")
            print(
                f"  Spectral Centroid: {orig_metrics['spectral']['centroid']:.0f} → "
                f"{proc_metrics['spectral']['centroid']:.0f} Hz"
            )
            print()

        # Category summary
        if cat_results:
            cat_snr_improvements = [r["improvements"]["snr_improvement"] for r in cat_results]  # type: ignore[index]
            cat_thd_changes = [r["improvements"]["thd_change"] for r in cat_results]  # type: ignore[index]

            results["per_category"][cat] = {
                "count": len(cat_results),
                "avg_snr_improvement": np.mean(cat_snr_improvements),
                "avg_thd_change": np.mean(cat_thd_changes),
                "min_snr_improvement": np.min(cat_snr_improvements),
                "max_snr_improvement": np.max(cat_snr_improvements),
            }

            print("Category Summary:")
            print(f"  Files processed: {len(cat_results)}")
            print(f"  Avg SNR improvement: {np.mean(cat_snr_improvements):+.2f} dB")
            print(f"  SNR range: {np.min(cat_snr_improvements):+.2f} to {np.max(cat_snr_improvements):+.2f} dB")
            print(f"  Avg THD change: {np.mean(cat_thd_changes):+.3f}%")

    # Global summary
    if snr_improvements:
        results["summary"] = {
            "total_files": total_files,
            "avg_snr_improvement": float(np.mean(snr_improvements)),
            "std_snr_improvement": float(np.std(snr_improvements)),
            "min_snr_improvement": float(np.min(snr_improvements)),
            "max_snr_improvement": float(np.max(snr_improvements)),
            "avg_thd_change": float(np.mean(thd_changes)),
            "std_thd_change": float(np.std(thd_changes)),
            "success_criteria": {
                "snr_target": 10.0,  # dB
                "snr_achieved": float(np.mean(snr_improvements)) >= 10.0,
                "thd_target": 1.0,  # % max increase
                "thd_achieved": abs(float(np.mean(thd_changes))) <= 1.0,
            },
        }

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total files validated: {total_files}")
    print(f"Average SNR improvement: {np.mean(snr_improvements):+.2f} ± {np.std(snr_improvements):.2f} dB")
    print(f"SNR range: {np.min(snr_improvements):+.2f} to {np.max(snr_improvements):+.2f} dB")
    print(f"Average THD change: {np.mean(thd_changes):+.3f} ± {np.std(thd_changes):.3f}%")
    print("\nSuccess Criteria:")
    print(
        f"  ✓ SNR improvement >10 dB: {'YES' if results['summary']['success_criteria']['snr_achieved'] else 'NO'} "
        f"({np.mean(snr_improvements):.1f} dB)"
    )
    print(
        f"  ✓ THD increase <1%: {'YES' if results['summary']['success_criteria']['thd_achieved'] else 'NO'} "
        f"({abs(np.mean(thd_changes)):.2f}%)"
    )
    print(f"\nResults saved to: {output_file}")
    print(f"{'=' * 80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run validation suite on original vs processed files")
    parser.add_argument("--library", type=Path, default=Path("test_library"), help="Path to test library directory")
    parser.add_argument(
        "--processed", type=Path, default=Path("test_library/aurik_processed"), help="Path to AURIK-processed files"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("validation_report.json"), help="Output file for validation report"
    )
    parser.add_argument(
        "--category",
        choices=["vinyl", "tape", "digital", "vocals"],
        help="Validate only specific category (default: all)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.library.exists():
        print(f"❌ Test library not found: {args.library}")
        return 1

    if not args.processed.exists():
        print(f"❌ Processed files not found: {args.processed}")
        print("   Run batch_process.py first to process test files")
        return 1

    # Run validation
    run_validation(args.library, args.processed, args.output, args.category)

    return 0


if __name__ == "__main__":
    sys.exit(main())
