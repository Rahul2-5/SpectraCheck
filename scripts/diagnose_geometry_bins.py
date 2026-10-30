"""Read-only geometry-bin matching analysis for a blocked metadata audit.

The command reads an existing ``active_manifest.csv`` (or the original audit
when explicitly requested), creates deterministic *diagnostic-only* cohorts by
matching short/long image-side bins and orientation-invariant aspect bins, and
re-runs the repository's existing ``aidetect.audit.shortcut_probe``.  It never
changes images/manifests, preprocessing, thresholds, or research splits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aidetect.audit import PROBE_COLUMNS, shortcut_probe  # noqa: E402
from diagnose_metadata_shortcut import (  # noqa: E402
    _balanced,
    _eligible,
    _load_active,
    _load_original,
)
from aidetect.io_utils import config_hash, load_config  # noqa: E402


THRESHOLD = 0.60


@dataclass(frozen=True)
class Strategy:
    """One explicit, reproducible resolution/aspect binning rule."""

    name: str
    resolution_mode: str
    resolution_bins: int | None = None
    resolution_pixels: int | None = None
    aspect_bins: int | None = None
    aspect_log2_width: float | None = None


STRATEGIES = (
    Strategy("pixel16_aspect005", "fixed", resolution_pixels=16, aspect_log2_width=0.05),
    Strategy("pixel32_aspect010", "fixed", resolution_pixels=32, aspect_log2_width=0.10),
    Strategy("pixel64_aspect020", "fixed", resolution_pixels=64, aspect_log2_width=0.20),
    Strategy("pixel128_aspect025", "fixed", resolution_pixels=128, aspect_log2_width=0.25),
    Strategy("pooled_quantile4_aspect4", "pooled_quantile", resolution_bins=4, aspect_bins=4),
    Strategy("pooled_quantile8_aspect8", "pooled_quantile", resolution_bins=8, aspect_bins=8),
)


def _strategy_record(strategy: Strategy) -> dict[str, Any]:
    return {
        "name": strategy.name,
        "resolution_mode": strategy.resolution_mode,
        "resolution_pixels": strategy.resolution_pixels,
        "resolution_bins": strategy.resolution_bins,
        "aspect_log2_width": strategy.aspect_log2_width,
        "aspect_bins": strategy.aspect_bins,
        "aspect_definition": "abs(log2(max(width,height)/min(width,height)))",
        "resolution_definition": "short_side and long_side independently",
    }


def _geometry_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["width"] = result["width"].astype(float)
    result["height"] = result["height"].astype(float)
    result["short_side"] = result[["width", "height"]].min(axis=1)
    result["long_side"] = result[["width", "height"]].max(axis=1)
    result["aspect_abs"] = result["long_side"] / result["short_side"]
    result["aspect_log2_abs"] = np.abs(np.log2(result["aspect_abs"]))
    if not np.isfinite(result[["short_side", "long_side", "aspect_abs", "aspect_log2_abs"]]).all().all():
        raise ValueError("Geometry metadata contains non-finite values")
    return result


def _quantile_edges(values: pd.Series, bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("Quantile bin count must be positive")
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        # An empty candidate can still be passed through every strategy.  A
        # single catch-all interval keeps bin construction total and lets the
        # caller record an empty cohort instead of aborting the run.
        return np.asarray([-np.inf, np.inf], dtype=float)
    raw = np.quantile(numeric.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    unique = np.unique(raw)
    if unique.size < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)
    return np.r_[-np.inf, unique[1:-1], np.inf]


def _fixed_bin(values: pd.Series, width: float) -> pd.Series:
    if width <= 0:
        raise ValueError("Fixed bin width must be positive")
    return np.floor(values / width).astype(int).astype(str)


def _quantile_bin(values: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = np.digitize(values.to_numpy(dtype=float), edges[1:-1], right=False)
    return pd.Series(labels.astype(int).astype(str), index=values.index)


def _apply_strategy(frame: pd.DataFrame, strategy: Strategy, pooled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = _geometry_values(frame)
    pooled_values = _geometry_values(pooled)
    if strategy.resolution_mode == "fixed":
        assert strategy.resolution_pixels is not None and strategy.aspect_log2_width is not None
        short_bin = _fixed_bin(result["short_side"], strategy.resolution_pixels)
        long_bin = _fixed_bin(result["long_side"], strategy.resolution_pixels)
        aspect_bin = _fixed_bin(result["aspect_log2_abs"], strategy.aspect_log2_width)
        parameters = _strategy_record(strategy)
    elif strategy.resolution_mode == "pooled_quantile":
        assert strategy.resolution_bins is not None and strategy.aspect_bins is not None
        short_bin = _quantile_bin(
            result["short_side"], _quantile_edges(pooled_values["short_side"], strategy.resolution_bins)
        )
        long_bin = _quantile_bin(
            result["long_side"], _quantile_edges(pooled_values["long_side"], strategy.resolution_bins)
        )
        aspect_bin = _quantile_bin(
            result["aspect_log2_abs"], _quantile_edges(pooled_values["aspect_log2_abs"], strategy.aspect_bins)
        )
        parameters = _strategy_record(strategy)
        parameters["pooled_rows_for_edges"] = int(len(pooled))
    else:
        raise ValueError(f"Unknown resolution mode: {strategy.resolution_mode}")
    result["short_bin"] = short_bin
    result["long_bin"] = long_bin
    result["aspect_bin"] = aspect_bin
    result["geometry_bin"] = (
        "s" + result["short_bin"] + "_l" + result["long_bin"] + "_a" + result["aspect_bin"]
    )
    return result, parameters


def _stable_rank(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def _match_cohort(frame: pd.DataFrame, strategy: Strategy, pooled: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned, parameters = _apply_strategy(frame, strategy, pooled)
    counts = binned.groupby(["geometry_bin", "label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in counts:
            counts[label] = 0
    counts = counts[[0, 1]].rename(columns={0: "real_count", 1: "ai_count"})
    counts["matched_per_class"] = counts.min(axis=1).astype(int)
    counts["real_loss"] = counts["real_count"] - counts["matched_per_class"]
    counts["ai_loss"] = counts["ai_count"] - counts["matched_per_class"]
    counts = counts.reset_index()

    selected: list[pd.DataFrame] = []
    for (geometry_bin, label), group in binned.groupby(["geometry_bin", "label"], sort=True):
        target = int(counts.loc[counts["geometry_bin"] == geometry_bin, "matched_per_class"].iloc[0])
        ranked = group.assign(_stable_rank=group["id"].astype(str).map(lambda value: _stable_rank(seed, value)))
        selected.append(ranked.sort_values(["_stable_rank", "id"]).head(target))
    cohort = pd.concat(selected, ignore_index=True) if selected else binned.iloc[0:0].copy()
    cohort = cohort.drop(columns=["_stable_rank"], errors="ignore").sort_values("id").reset_index(drop=True)
    counts["strategy"] = strategy.name
    counts["parameters"] = json.dumps(parameters, sort_keys=True)
    counts["balanced_bin"] = counts["real_count"] == counts["ai_count"]
    counts["selected_rows"] = 2 * counts["matched_per_class"]
    counts = counts[[
        "strategy", "geometry_bin", "real_count", "ai_count", "matched_per_class",
        "selected_rows", "real_loss", "ai_loss", "balanced_bin", "parameters",
    ]].sort_values("geometry_bin").reset_index(drop=True)
    return cohort, counts


def _distribution_rows(frame: pd.DataFrame, strategy: str, population: str) -> pd.DataFrame:
    values = _geometry_values(frame)
    rows: list[dict[str, Any]] = []
    features = ("width", "height", "short_side", "long_side", "aspect_abs", "aspect_ratio")
    for label, class_name in ((0, "real"), (1, "ai")):
        subset = values.loc[values["label"] == label]
        for feature in features:
            if feature in subset:
                numeric = pd.to_numeric(subset[feature], errors="coerce")
            else:
                numeric = pd.Series(index=subset.index, dtype=float)
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            finite = numeric.dropna().to_numpy(dtype=float)
            if finite.size == 0:
                quantiles = np.full(7, np.nan, dtype=float)
                status = "empty_cohort" if len(subset) == 0 else "empty_numeric"
            else:
                quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                status = "partial_missing" if len(finite) < len(numeric) else "ok"
            rows.append({
                "strategy": strategy,
                "population": population,
                "label": label,
                "class_name": class_name,
                "feature": feature,
                "count": int(len(subset)),
                "numeric_count": int(finite.size),
                "missing_count": int(len(numeric) - finite.size),
                "status": status,
                "min": float(np.min(finite)) if finite.size else np.nan,
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "q99": float(quantiles[6]),
                "max": float(np.max(finite)) if finite.size else np.nan,
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size else np.nan),
            })
    return pd.DataFrame(rows)


def _probe_result(strategy: str, cohort: pd.DataFrame, *, seed: int, cv_folds: int,
                  tree_depth: int, logistic_max_iter: int) -> pd.DataFrame:
    if "label" in cohort:
        counts = cohort["label"].value_counts().to_dict()
        real_rows = int(counts.get(0, 0))
        ai_rows = int(counts.get(1, 0))
        min_class_count = min(real_rows, ai_rows)
    else:
        real_rows = ai_rows = min_class_count = 0
    base = {
        "strategy": strategy,
        "rows": int(len(cohort)),
        "real_rows": real_rows,
        "ai_rows": ai_rows,
        "threshold": THRESHOLD,
        "cv_folds": cv_folds,
        "seed": seed,
    }
    if "label" not in cohort:
        reason = "missing_label_column"
        status = "error"
    elif cohort.empty:
        reason = "empty_cohort"
        status = "not_run"
    elif min_class_count < cv_folds:
        reason = "class_count_below_cv_folds"
        status = "not_run"
    else:
        reason = ""
        status = "ready"
    if status != "ready":
        return pd.DataFrame([
            {**base, "model": model, "probe_status": status, "reason": reason,
             "mean_accuracy": np.nan, "fold_accuracies": "", "passes_threshold": False}
            for model in ("logistic_regression", "decision_tree")
        ])
    try:
        result = shortcut_probe(
            cohort,
            seed,
            cv_folds=cv_folds,
            tree_depth=tree_depth,
            logistic_max_iter=logistic_max_iter,
        )
    except Exception as exc:  # Preserve a diagnostic record rather than hiding a failed candidate.
        return pd.DataFrame([
            {**base, "model": model, "probe_status": "error", "reason": f"{type(exc).__name__}: {exc}",
             "mean_accuracy": np.nan, "fold_accuracies": "", "passes_threshold": False}
            for model in ("logistic_regression", "decision_tree")
        ])
    rows = []
    for model, values in result["models"].items():
        mean = float(values["mean_accuracy"])
        rows.append({
            **base,
            "model": model,
            "probe_status": "ran",
            "reason": "",
            "mean_accuracy": mean,
            "fold_accuracies": json.dumps(values["fold_accuracies"]),
            "passes_threshold": bool(mean <= THRESHOLD),
        })
    return pd.DataFrame(rows)


def _summary_row(strategy: str, population: str, source: pd.DataFrame, cohort: pd.DataFrame,
                 counts: pd.DataFrame, image_size: int) -> dict[str, Any]:
    eligible = _eligible(source, image_size)
    real_available = int((eligible["label"] == 0).sum())
    ai_available = int((eligible["label"] == 1).sum())
    real_selected = int((cohort["label"] == 0).sum())
    ai_selected = int((cohort["label"] == 1).sum())
    return {
        "strategy": strategy,
        "population": population,
        "all_rows": int(len(source)),
        "eligible_rows": int(len(eligible)),
        "undersized_real": int(((source["label"] == 0) & ((source["width"] < image_size) | (source["height"] < image_size))).sum()),
        "undersized_ai": int(((source["label"] == 1) & ((source["width"] < image_size) | (source["height"] < image_size))).sum()),
        "eligible_real": real_available,
        "eligible_ai": ai_available,
        "selected_real": real_selected,
        "selected_ai": ai_selected,
        "selected_total": len(cohort),
        "real_excluded_after_eligibility": real_available - real_selected,
        "ai_excluded_after_eligibility": ai_available - ai_selected,
        "balanced": real_selected == ai_selected,
        "class_balance_difference": real_selected - ai_selected,
        "real_retention_of_eligible": real_selected / real_available if real_available else np.nan,
        "ai_retention_of_eligible": ai_selected / ai_available if ai_available else np.nan,
        "geometry_bins": int(len(counts)),
        "bins_with_both_classes": int((counts["balanced_bin"] & (counts["matched_per_class"] > 0)).sum()),
    }


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run_geometry_analysis(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    scope: str = "active",
    image_size: int = 256,
    seed: int = 42,
    cv_folds: int = 5,
    tree_depth: int = 3,
    logistic_max_iter: int = 2000,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all bin strategies and write only to a separate output directory."""
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Diagnostic output must not be inside data_root")
    if scope not in {"active", "original"}:
        raise ValueError("scope must be active or original")

    active, active_input = _load_active(root)
    original, original_input = _load_original(root, reinspect=False)
    if scope == "active":
        if active is None:
            raise FileNotFoundError("Requested active scope but data_root/audit/active_manifest.csv is missing")
        source = active
        source_input = active_input
    else:
        source = original
        source_input = original_input

    eligible = _eligible(source, image_size)

    all_counts: list[pd.DataFrame] = []
    all_distributions: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    all_probes: list[pd.DataFrame] = []
    all_ids: list[pd.DataFrame] = []
    strategy_parameters: list[dict[str, Any]] = []

    eligible_counts = eligible["label"].value_counts().to_dict()
    if min(int(eligible_counts.get(0, 0)), int(eligible_counts.get(1, 0))) >= cv_folds:
        baseline = _balanced(eligible, seed)
    else:
        # Preserve a diagnostic record for undersized/one-class inputs; the
        # probe function will mark it not_run instead of aborting all bins.
        baseline = eligible.copy()
    all_probes.append(_probe_result(
        "active_unmatched" if scope == "active" else "original_unmatched",
        baseline,
        seed=seed,
        cv_folds=cv_folds,
        tree_depth=tree_depth,
        logistic_max_iter=logistic_max_iter,
    ))
    all_distributions.append(_distribution_rows(source, "unmatched", "all"))
    all_distributions.append(_distribution_rows(eligible, "unmatched", "eligible"))

    for strategy in STRATEGIES:
        cohort, counts = _match_cohort(eligible, strategy, eligible, seed)
        strategy_parameters.append(_strategy_record(strategy))
        all_counts.append(counts)
        all_summaries.append(_summary_row(strategy.name, "eligible", source, cohort, counts, image_size))
        all_probes.append(_probe_result(
            strategy.name,
            cohort,
            seed=seed,
            cv_folds=cv_folds,
            tree_depth=tree_depth,
            logistic_max_iter=logistic_max_iter,
        ))
        all_distributions.append(_distribution_rows(cohort, strategy.name, "candidate"))
        if not cohort.empty:
            all_ids.append(cohort[["id", "label", "geometry_bin"]].assign(strategy=strategy.name))

    counts = pd.concat(all_counts, ignore_index=True)
    summaries = pd.DataFrame(all_summaries)
    probes = pd.concat(all_probes, ignore_index=True)
    distributions = pd.concat(all_distributions, ignore_index=True)
    ids = pd.concat(all_ids, ignore_index=True) if all_ids else pd.DataFrame(columns=["strategy", "id", "label", "geometry_bin"])

    probe_accuracy = probes.loc[probes["probe_status"] == "ran"].pivot_table(
        index="strategy", columns="model", values="mean_accuracy", aggfunc="first"
    )
    probe_status = probes.groupby("strategy")["probe_status"].agg(lambda values: ",".join(sorted(set(values))))
    logistic_accuracy = probe_accuracy["logistic_regression"] if "logistic_regression" in probe_accuracy else pd.Series(dtype=float)
    tree_accuracy = probe_accuracy["decision_tree"] if "decision_tree" in probe_accuracy else pd.Series(dtype=float)
    summaries["logistic_accuracy"] = summaries["strategy"].map(logistic_accuracy)
    summaries["tree_accuracy"] = summaries["strategy"].map(tree_accuracy)
    summaries["max_probe_accuracy"] = summaries[["logistic_accuracy", "tree_accuracy"]].max(axis=1)
    summaries["probe_status"] = summaries["strategy"].map(probe_status).fillna("not_run")
    summaries["both_models_pass"] = (
        (summaries["probe_status"] == "ran")
        & (summaries["logistic_accuracy"] <= THRESHOLD)
        & (summaries["tree_accuracy"] <= THRESHOLD)
    )
    summaries["candidate_assessment"] = np.select(
        [summaries["probe_status"] != "ran", summaries["both_models_pass"]],
        ["not_validated", "below_threshold_diagnostic_candidate"],
        default="blocked_by_metadata_probe",
    )

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "distributions": output / "geometry_bin_distributions.csv",
        "bin_counts": output / "geometry_bin_counts.csv",
        "summary": output / "geometry_bin_summary.csv",
        "probes": output / "geometry_bin_probe_results.csv",
        "cohort_ids": output / "geometry_bin_cohort_ids.csv",
        "run_json": output / "geometry_bin_run.json",
        "report": output / "geometry_bin_matching_summary.md",
    }
    _write(distributions, paths["distributions"])
    _write(counts, paths["bin_counts"])
    _write(summaries, paths["summary"])
    _write(probes, paths["probes"])
    _write(ids, paths["cohort_ids"])

    config_digest = None
    if config_path is not None and Path(config_path).is_file():
        config_digest = config_hash(load_config(config_path))
    record = {
        "data_root": str(root),
        "output_dir": str(output),
        "scope": scope,
        "source_input": source_input,
        "original_input": original_input,
        "active_input": active_input,
        "image_size": image_size,
        "seed": seed,
        "cv_folds": cv_folds,
        "tree_depth": tree_depth,
        "logistic_max_iter": logistic_max_iter,
        "threshold": THRESHOLD,
        "threshold_changed": False,
        "preprocessing_changed": False,
        "manifests_modified": False,
        "images_modified": False,
        "splits_created": False,
        "strategies": strategy_parameters,
        "config_path": str(Path(config_path).resolve()) if config_path is not None else None,
        "config_sha256": config_digest,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["run_json"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    probe_view = probes[["strategy", "model", "probe_status", "mean_accuracy", "passes_threshold", "rows", "real_rows", "ai_rows"]]
    summary_view = summaries[[
        "strategy", "eligible_real", "eligible_ai", "selected_real", "selected_ai", "selected_total",
        "real_excluded_after_eligibility", "ai_excluded_after_eligibility", "real_retention_of_eligible",
        "ai_retention_of_eligible", "balanced", "logistic_accuracy", "tree_accuracy",
        "max_probe_accuracy", "both_models_pass", "candidate_assessment",
    ]]
    report_lines = [
        "# Geometry-bin matching analysis",
        "",
        "Status: **completed**",
        "",
        "This is a read-only diagnostic. It does not modify images, manifests, preprocessing, thresholds, or research splits.",
        "",
        f"- Scope: `{scope}`",
        f"- Input: `{source_input}`",
        f"- Eligibility: width and height >= {image_size}",
        f"- Seed/folds: {seed}/{cv_folds}",
        f"- Shortcut threshold: {THRESHOLD:.2f}, unchanged",
        "- `make_splits.py`: not run",
        "",
        "## Candidate retention and exclusions",
        "",
        "Each candidate retains `min(real_count, ai_count)` rows within every geometry bin. This guarantees exact class balance within the diagnostic cohort; it does not establish research validity.",
        "",
        "```text",
        summary_view.to_string(index=False),
        "```",
        "",
        "## Existing metadata probe results",
        "",
        "A candidate is only provisionally below the configured gate when both actual probe rows have `probe_status=ran` and `passes_threshold=True`. `not_run` and `error` are not passes.",
        "",
        "```text",
        probe_view.to_string(index=False),
        "```",
        "",
        "## Strategy definitions",
        "",
        "- `pixel16_aspect005`: 16-pixel short/long-side bins and 0.05-wide absolute log2 aspect bins.",
        "- `pixel32_aspect010`: 32-pixel resolution bins and 0.10-wide aspect bins.",
        "- `pixel64_aspect020`: 64-pixel resolution bins and 0.20-wide aspect bins.",
        "- `pixel128_aspect025`: 128-pixel resolution bins and 0.25-wide aspect bins.",
        "- `pooled_quantile4_aspect4`: four pooled, label-independent quantile bins for each geometry dimension.",
        "- `pooled_quantile8_aspect8`: eight pooled, label-independent quantile bins for each geometry dimension.",
        "",
        "Aspect bins use `abs(log2(max(width,height)/min(width,height)))`, so portrait and landscape forms of the same geometry share a bin.",
        "",
        "## Outputs",
        "",
    ]
    report_lines.extend(f"- `{path.name}`" for path in paths.values())
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "A strategy with meaningful retention and both probe means <= 0.60 is a diagnostic candidate for review only. It is not an approved dataset change, and this command does not alter the active cohort or create splits.",
    ])
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("active", "original"), default="active")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_geometry_analysis(
        args.data_root,
        args.output_dir,
        scope=args.scope,
        image_size=args.image_size,
        seed=args.seed,
        cv_folds=args.cv_folds,
        tree_depth=args.tree_depth,
        logistic_max_iter=args.logistic_max_iter,
        config_path=args.config,
    )
    print(json.dumps({
        "status": "completed",
        "output_dir": result["output_dir"],
        "splits_created": result["splits_created"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
