"""Read-only metadata shortcut diagnostics for an existing acquisition.

This command reads the original main collection and, when present, the audited
active manifest.  It writes diagnostics to a separate output directory only:

* per-class numeric distributions;
* exact dimension frequencies;
* full, single-feature, and leave-one-feature-out probe results;
* feature-contribution rankings; and
* deterministic geometry-matching counts.

It never calls the audit workflow, normalization, split creation, feature
extraction, or model training.  It never writes under ``data_root``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aidetect.audit import audit_table  # noqa: E402
from aidetect.io_utils import config_hash, load_config, sha256_of_file  # noqa: E402


METADATA_FEATURES = (
    "is_jpeg",
    "jpeg_quality",
    "width",
    "height",
    "n_bytes",
    "aspect_ratio",
)
REQUIRED_COLUMNS = {"id", "rel_path", "label", "sha256"}
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    return pd.read_csv(path, dtype={"id": str, "rel_path": str, "sha256": str})


def _validate_manifest(frame: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{name} has missing or duplicate IDs")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{name} has labels outside real=0 and AI=1")
    if frame["sha256"].isna().any():
        raise ValueError(f"{name} has missing image hashes")


def _assert_original_audit_matches_manifest(
    manifest: pd.DataFrame, audited: pd.DataFrame, audit_path: Path
) -> None:
    _validate_manifest(audited, str(audit_path))
    left = manifest[["id", "label", "sha256"]].copy().sort_values("id").reset_index(drop=True)
    right = audited[["id", "label", "sha256"]].copy().sort_values("id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(
            f"{audit_path} does not match main/manifest.csv by id, label, and sha256"
        )


def _load_original(data_root: Path, *, reinspect: bool) -> tuple[pd.DataFrame, str]:
    main_root = data_root / "main"
    manifest_path = main_root / "manifest.csv"
    manifest = _read_table(manifest_path)
    _validate_manifest(manifest, str(manifest_path))
    audit_path = data_root / "audit" / "original_audit.csv"

    if audit_path.is_file() and not reinspect:
        audited = _read_table(audit_path)
        _assert_original_audit_matches_manifest(manifest, audited, audit_path)
        frame = audited
        source = audit_path.as_posix()
    else:
        frame = audit_table(manifest, main_root)
        source = "audit_table(main/manifest.csv, data_root/main)"

    _validate_measured(frame, "original measurements")
    return frame, source


def _load_active(data_root: Path) -> tuple[pd.DataFrame | None, str | None]:
    path = data_root / "audit" / "active_manifest.csv"
    if not path.is_file():
        return None, None
    frame = _read_table(path)
    _validate_measured(frame, str(path))
    result_path = data_root / "audit" / "audit_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = result.get("active_manifest_sha256")
        if expected and expected != sha256_of_file(path):
            raise ValueError("active_manifest.csv does not match audit_result.json")
    return frame, path.as_posix()


def _validate_measured(frame: pd.DataFrame, name: str) -> None:
    _validate_manifest(frame, name)
    missing = set(METADATA_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing measured metadata: {sorted(missing)}")
    numeric = frame[list(METADATA_FEATURES)].copy()
    numeric["jpeg_quality"] = numeric["jpeg_quality"].fillna(-1.0)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite metadata values")


def _eligible(frame: pd.DataFrame, image_size: int) -> pd.DataFrame:
    if image_size < 1:
        raise ValueError("image_size must be positive")
    return frame.loc[
        (frame["width"].astype(float) >= image_size)
        & (frame["height"].astype(float) >= image_size)
    ].copy()


def _balanced(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = frame["label"].value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError("Diagnostic probe requires both labels")
    ordered = frame.sort_values("id").reset_index(drop=True)
    real = ordered.loc[ordered["label"] == 0]
    ai = ordered.loc[ordered["label"] == 1]
    if real.empty or ai.empty:
        raise ValueError("Diagnostic probe requires non-empty real and AI classes")
    size = min(len(real), len(ai))
    chosen_real = real.sample(n=size, random_state=seed)
    chosen_ai = ai.sample(n=size, random_state=seed)
    return pd.concat([chosen_real, chosen_ai], ignore_index=True).sort_values("id").reset_index(drop=True)


def _probe_models(
    frame: pd.DataFrame,
    columns: Iterable[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    tree_depth: int,
    logistic_max_iter: int,
) -> dict[str, Any]:
    selected = list(columns)
    values = frame[selected].copy()
    if "jpeg_quality" in values:
        values["jpeg_quality"] = values["jpeg_quality"].fillna(-1.0)
    x = values.to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("Probe metadata is non-finite")
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=logistic_max_iter, random_state=seed),
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=tree_depth, random_state=seed),
    }
    result: dict[str, Any] = {}
    for name, model in models.items():
        scores = cross_val_score(model, x, y, cv=folds, scoring="accuracy", error_score="raise")
        result[name] = {
            "mean_accuracy": float(scores.mean()),
            "fold_accuracies": [float(value) for value in scores],
        }
    return result


def _ablation(frame: pd.DataFrame, *, seed: int, folds_count: int, tree_depth: int,
              logistic_max_iter: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame["label"].value_counts()
    if counts.min() < folds_count:
        raise ValueError("Each class needs at least cv_folds rows for ablation")
    splitter = StratifiedKFold(n_splits=folds_count, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(frame)), frame["label"].to_numpy()))
    rows: list[dict[str, Any]] = []
    specifications: list[tuple[str, list[str], str | None]] = [
        ("full", list(METADATA_FEATURES), None),
    ]
    specifications.extend(
        ("single_feature", [feature], feature) for feature in METADATA_FEATURES
    )
    specifications.extend(
        ("leave_one_out", [feature for feature in METADATA_FEATURES if feature != removed], removed)
        for removed in METADATA_FEATURES
    )
    for method, columns, feature in specifications:
        scores = _probe_models(
            frame,
            columns,
            folds,
            seed=seed,
            tree_depth=tree_depth,
            logistic_max_iter=logistic_max_iter,
        )
        for model_name, result in scores.items():
            rows.append({
                "method": method,
                "feature": feature or "__all__",
                "features_used": json.dumps(columns),
                "model": model_name,
                "mean_accuracy": result["mean_accuracy"],
                "fold_accuracies": json.dumps(result["fold_accuracies"]),
                "cv_folds": folds_count,
                "seed": seed,
            })
    ablation = pd.DataFrame(rows)
    full = ablation.loc[ablation["method"] == "full"].set_index("model")["mean_accuracy"]
    ranking_rows: list[dict[str, Any]] = []
    for model_name in sorted(full.index):
        full_score = float(full.loc[model_name])
        single = ablation.loc[
            (ablation["model"] == model_name) & (ablation["method"] == "single_feature")
        ].set_index("feature")["mean_accuracy"]
        loo = ablation.loc[
            (ablation["model"] == model_name) & (ablation["method"] == "leave_one_out")
        ].set_index("feature")["mean_accuracy"]
        ranking = pd.DataFrame({
            "model": model_name,
            "feature": list(METADATA_FEATURES),
            "full_accuracy": full_score,
            "single_feature_accuracy": [float(single[feature]) for feature in METADATA_FEATURES],
            "leave_one_out_accuracy": [float(loo[feature]) for feature in METADATA_FEATURES],
        })
        ranking["single_feature_excess_over_chance"] = ranking["single_feature_accuracy"] - 0.5
        ranking["leave_one_out_drop_from_full"] = (
            ranking["full_accuracy"] - ranking["leave_one_out_accuracy"]
        )
        ranking["rank_by_leave_one_out_drop"] = (
            ranking["leave_one_out_drop_from_full"].rank(method="min", ascending=False).astype(int)
        )
        ranking_rows.extend(ranking.to_dict("records"))
    return ablation, pd.DataFrame(ranking_rows).sort_values(
        ["model", "rank_by_leave_one_out_drop", "feature"]
    ).reset_index(drop=True)


def _distribution_rows(frame: pd.DataFrame, source_scope: str, population: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, class_name in ((0, "real"), (1, "ai")):
        subset = frame.loc[frame["label"] == label]
        for feature in METADATA_FEATURES:
            values = pd.to_numeric(subset[feature], errors="coerce")
            missing = int(values.isna().sum())
            finite = values.dropna().to_numpy(dtype=float)
            row: dict[str, Any] = {
                "source_scope": source_scope,
                "population": population,
                "label": label,
                "class_name": class_name,
                "feature": feature,
                "count": int(len(subset)),
                "missing_count": missing,
                "unique_count": int(values.nunique(dropna=True)),
            }
            if finite.size:
                quantiles = np.quantile(finite, QUANTILES)
                row.update({
                    "min": float(np.min(finite)),
                    "q01": float(quantiles[0]),
                    "q05": float(quantiles[1]),
                    "q25": float(quantiles[2]),
                    "median": float(quantiles[3]),
                    "q75": float(quantiles[4]),
                    "q95": float(quantiles[5]),
                    "q99": float(quantiles[6]),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                })
            else:
                row.update({key: None for key in (
                    "min", "q01", "q05", "q25", "median", "q75", "q95", "q99",
                    "max", "mean", "std",
                )})
            rows.append(row)
    return rows


def _dimension_frequency_rows(frame: pd.DataFrame, source_scope: str, population: str) -> pd.DataFrame:
    grouped = (
        frame.assign(width=frame["width"].astype(int), height=frame["height"].astype(int))
        .groupby(["width", "height", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pivot = grouped.pivot_table(
        index=["width", "height"], columns="label", values="count", fill_value=0, aggfunc="sum"
    ).reset_index()
    pivot.columns = ["width", "height", *[f"label_{int(column)}_count" for column in pivot.columns[2:]]]
    for column in ("label_0_count", "label_1_count"):
        if column not in pivot:
            pivot[column] = 0
    pivot["source_scope"] = source_scope
    pivot["population"] = population
    pivot["total_count"] = pivot["label_0_count"] + pivot["label_1_count"]
    pivot["ai_fraction"] = np.where(
        pivot["total_count"] > 0, pivot["label_1_count"] / pivot["total_count"], np.nan
    )
    return pivot[[
        "source_scope", "population", "width", "height", "label_0_count", "label_1_count",
        "total_count", "ai_fraction",
    ]].sort_values(["source_scope", "population", "width", "height"]).reset_index(drop=True)


def _matching_rows(frame: pd.DataFrame, source_scope: str, population: str, scheme: str) -> pd.DataFrame:
    working = frame.copy()
    if scheme == "ordered_width_height":
        working["geometry_key"] = working["width"].astype(int).astype(str) + "x" + working["height"].astype(int).astype(str)
    elif scheme == "orientation_invariant_dimensions":
        low = working[["width", "height"]].min(axis=1).astype(int)
        high = working[["width", "height"]].max(axis=1).astype(int)
        working["geometry_key"] = low.astype(str) + "x" + high.astype(str)
    else:
        raise ValueError(f"Unknown geometry matching scheme: {scheme}")
    counts = working.groupby(["geometry_key", "label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in counts:
            counts[label] = 0
    result = counts[[0, 1]].rename(columns={0: "real_count", 1: "ai_count"}).reset_index()
    result["matched_per_class"] = result[["real_count", "ai_count"]].min(axis=1).astype(int)
    result["real_loss"] = result["real_count"] - result["matched_per_class"]
    result["ai_loss"] = result["ai_count"] - result["matched_per_class"]
    result["source_scope"] = source_scope
    result["population"] = population
    result["scheme"] = scheme
    return result[[
        "source_scope", "population", "scheme", "geometry_key", "real_count", "ai_count",
        "matched_per_class", "real_loss", "ai_loss",
    ]].sort_values("geometry_key").reset_index(drop=True)


def _matching_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["source_scope", "population", "scheme"], as_index=False).agg(
        real_available=("real_count", "sum"),
        ai_available=("ai_count", "sum"),
        matched_per_class=("matched_per_class", "sum"),
        real_loss=("real_loss", "sum"),
        ai_loss=("ai_loss", "sum"),
    )
    grouped["matched_total"] = 2 * grouped["matched_per_class"]
    grouped["post_match_real"] = grouped["matched_per_class"]
    grouped["post_match_ai"] = grouped["matched_per_class"]
    grouped["post_match_balance"] = grouped["post_match_real"] == grouped["post_match_ai"]
    grouped["real_retention"] = np.where(
        grouped["real_available"] > 0,
        grouped["post_match_real"] / grouped["real_available"],
        np.nan,
    )
    grouped["ai_retention"] = np.where(
        grouped["ai_available"] > 0,
        grouped["post_match_ai"] / grouped["ai_available"],
        np.nan,
    )
    return grouped


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 20) -> str:
    if frame.empty:
        return "(no rows)"
    return "```text\n" + frame[columns].head(rows).to_string(index=False) + "\n```"


def run_diagnostic(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    image_size: int = 256,
    seed: int = 42,
    cv_folds: int = 5,
    tree_depth: int = 3,
    logistic_max_iter: int = 2000,
    reinspect_original: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run diagnostics without writing to ``data_root`` or changing manifests."""
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Diagnostic output must not be inside data_root")
    if cv_folds < 2 or tree_depth < 1 or logistic_max_iter < 1:
        raise ValueError("Invalid probe settings")

    original, original_input = _load_original(root, reinspect=reinspect_original)
    active, active_input = _load_active(root)
    scopes: list[tuple[str, pd.DataFrame]] = [("original", original)]
    if active is not None:
        scopes.append(("active", active))

    distribution_rows: list[dict[str, Any]] = []
    dimension_frames: list[pd.DataFrame] = []
    ablation_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    matching_frames: list[pd.DataFrame] = []
    probe_summary: dict[str, Any] = {}

    for source_scope, frame in scopes:
        for population, population_frame in (("all", frame), ("eligible", _eligible(frame, image_size))):
            distribution_rows.extend(_distribution_rows(population_frame, source_scope, population))
            dimension_frames.append(_dimension_frequency_rows(population_frame, source_scope, population))
        eligible = _eligible(frame, image_size)
        balanced = _balanced(eligible, seed)
        ablation, ranking = _ablation(
            balanced,
            seed=seed,
            folds_count=cv_folds,
            tree_depth=tree_depth,
            logistic_max_iter=logistic_max_iter,
        )
        ablation.insert(0, "source_scope", source_scope)
        ranking.insert(0, "source_scope", source_scope)
        ablation_frames.append(ablation)
        ranking_frames.append(ranking)
        probe_summary[source_scope] = {
            "all_rows": int(len(frame)),
            "eligible_rows": int(len(eligible)),
            "balanced_rows": int(len(balanced)),
            "class_counts_before_balance": {
                str(key): int(value) for key, value in eligible["label"].value_counts().to_dict().items()
            },
            "ablation": ablation.to_dict("records"),
        }
        for population, population_frame in (("all", frame), ("eligible", eligible)):
            for scheme in ("ordered_width_height", "orientation_invariant_dimensions"):
                matching_frames.append(_matching_rows(population_frame, source_scope, population, scheme))

    distributions = pd.DataFrame(distribution_rows)
    dimensions = pd.concat(dimension_frames, ignore_index=True)
    ablations = pd.concat(ablation_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    matching_by_key = pd.concat(matching_frames, ignore_index=True)
    matching_summary = _matching_summary(matching_by_key)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "distributions": output / "metadata_distributions.csv",
        "dimensions": output / "dimension_frequencies.csv",
        "ablation": output / "metadata_ablation.csv",
        "ranking": output / "metadata_feature_ranking.csv",
        "matching_by_key": output / "geometry_matching_by_key.csv",
        "matching_summary": output / "geometry_matching_summary.csv",
        "run_json": output / "diagnostic_run.json",
        "report": output / "metadata_shortcut_diagnostic.md",
    }
    _write_csv(distributions, paths["distributions"])
    _write_csv(dimensions, paths["dimensions"])
    _write_csv(ablations, paths["ablation"])
    _write_csv(rankings, paths["ranking"])
    _write_csv(matching_by_key, paths["matching_by_key"])
    _write_csv(matching_summary, paths["matching_summary"])

    config_digest = None
    if config_path is not None and Path(config_path).is_file():
        config = load_config(config_path)
        config_digest = config_hash(config)
    record = {
        "data_root": str(root),
        "output_dir": str(output),
        "original_input": original_input,
        "active_input": active_input,
        "image_size": image_size,
        "seed": seed,
        "cv_folds": cv_folds,
        "tree_depth": tree_depth,
        "logistic_max_iter": logistic_max_iter,
        "config_path": str(Path(config_path).resolve()) if config_path is not None else None,
        "config_sha256": config_digest,
        "threshold_not_modified": 0.60,
        "writes_outside_data_root_only": True,
        "make_splits_called": False,
        "probe_summary": probe_summary,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["run_json"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    full_rows = ablations[ablations["method"] == "full"].copy()
    top_rows = rankings.loc[rankings["rank_by_leave_one_out_drop"] == 1]
    report_lines = [
        "# Metadata shortcut diagnostic",
        "",
        "Status: **completed**",
        "",
        "This is a read-only diagnostic. It does not modify original images, manifests, preprocessing, thresholds, or research splits.",
        "",
        f"- Data root: `{root}`",
        f"- Original input: `{original_input}`",
        f"- Active input: `{active_input or 'not present'}`",
        f"- Eligibility rule: width and height >= {image_size}",
        f"- Probe seed/folds: {seed}/{cv_folds}",
        "- Probe threshold: 0.60, unchanged",
        "- `make_splits.py` called: no",
        "",
        "## Probe results",
        "",
        _markdown_table(full_rows, ["source_scope", "model", "mean_accuracy", "fold_accuracies"]),
        "",
        "The ablation CSV contains full, single-feature, and leave-one-feature-out results. The ranking uses the drop in full-probe accuracy after removing each feature; correlated features can make this measure non-unique.",
        "",
        "## Top leave-one-out contributors",
        "",
        _markdown_table(top_rows, ["source_scope", "model", "feature", "single_feature_accuracy", "leave_one_out_drop_from_full"]),
        "",
        "## Geometry matching",
        "",
        "Matching retains the minimum real/AI count within each geometry key, so the resulting cohort is exactly class-balanced. `ordered_width_height` is strict; `orientation_invariant_dimensions` uses min(width,height) × max(width,height).",
        "",
        _markdown_table(matching_summary, ["source_scope", "population", "scheme", "real_available", "ai_available", "matched_per_class", "matched_total", "real_loss", "ai_loss", "real_retention", "ai_retention"]),
        "",
        "## Outputs",
        "",
    ]
    for key, path in paths.items():
        report_lines.append(f"- `{path.name}`")
    report_lines.extend([
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python scripts/diagnose_metadata_shortcut.py --data-root <DATA_ROOT> --output-dir <OUTPUT_DIR> --config config.yaml",
        "```",
        "",
        "Use `--reinspect-original` to re-read original image bytes and recompute measurements instead of reusing `audit/original_audit.csv`.",
        "",
        "## Interpretation limits",
        "",
        "This diagnostic identifies metadata predictiveness and estimates deterministic geometry-matched sample sizes. It does not alter the audit gate, prove that a dataset is unbiased, or authorize split creation.",
    ])
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
    parser.add_argument("--reinspect-original", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_diagnostic(
        args.data_root,
        args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
        cv_folds=args.cv_folds,
        tree_depth=args.tree_depth,
        logistic_max_iter=args.logistic_max_iter,
        reinspect_original=args.reinspect_original,
        config_path=args.config,
    )
    print(json.dumps({
        "status": "completed",
        "output_dir": result["output_dir"],
        "make_splits_called": result["make_splits_called"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
