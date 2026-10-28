
"""Phase 2 orchestration: audit/remediate, then freeze splits only on a pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aidetect.audit import audit_table, describe_file, duplicate_groups, normalise_format, shortcut_probe
from aidetect.data import _atomic_json, _save_original
from aidetect.io_utils import config_hash, sha256_of_file
from aidetect.splits import balance_classes, freeze_splits, make_group_splits, validate_splits


def _summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    formats = df.groupby(["label", "format"]).size().unstack(fill_value=0)
    summary = df.groupby("label").agg(count=("id", "size"), median_width=("width", "median"),
        median_height=("height", "median"), jpeg_count=("jpeg_quality", "count"),
        median_jpeg_quality=("jpeg_quality", "median"),
        max_quality_table_mae=("jpeg_quality_table_mae", "max"))
    return formats, summary


def _write_audit_report(original: pd.DataFrame, before: pd.DataFrame, active: pd.DataFrame,
                        result: dict, report_dir: Path, settings: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    figures = report_dir / "figures"
    figures.mkdir(exist_ok=True)
    text = ["# Dataset audit report", "", f"Status: **{result['status']}**",
        f"Config SHA-256: `{result['config_hash']}`", "",
        "Probe rule: BOTH metadata-only CV mean accuracies must be <= "
        f"{result['threshold']:.2f}. Probes use balanced eligible classes; source splits are not final experiment splits.",
        "JPEG quality is a nearest-standard quantization-table estimate; a residual above zero means the match is approximate.",
        "Labels: real = 0, AI-generated = 1. Originals remain unchanged on disk.",
        "Passing this probe does not prove all confounding is removed.", ""]
    for name, frame in (("original", original), ("before", before), ("active", active)):
        formats, summary = _summary_tables(frame)
        formats.to_csv(report_dir / f"audit_{name}_formats.csv")
        summary.to_csv(report_dir / f"audit_{name}_summary.csv")
        text.extend([f"## {name.capitalize()} files", "", "```text", formats.to_string(),
                     "", summary.to_string(), "```", ""])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for label, title in ((0, "Real"), (1, "AI")):
            values = frame[frame["label"] == label]
            axes[0].hist(values[["width", "height"]].min(axis=1), bins=settings["histogram_bins"], alpha=.5, label=title)
            quality = values["jpeg_quality"].dropna()
            if not quality.empty:
                axes[1].hist(quality, bins=settings["histogram_bins"], alpha=.5, label=title)
        axes[0].set(xlabel="Minimum original side (pixels)", ylabel="Images", title="Image size")
        axes[1].set(xlabel="Estimated JPEG quality", ylabel="Images", title="JPEG quality (PNG omitted)")
        for axis in axes:
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        fig.tight_layout()
        fig.savefig(figures / f"audit_{name}.png", dpi=settings["figure_dpi"])
        plt.close(fig)
        text.extend([f"![{name} metadata](figures/audit_{name}.png)", ""])
    text.extend(["## Decisions and counts", "", "```json", json.dumps(result, indent=2, sort_keys=True),
                 "```", "", "If blocked, stop. Do not weaken the threshold or create splits to bypass the result.",
                 "No frequency features or classifier test-set performance were inspected.", ""])
    (report_dir / "audit_report.md").write_text("\n".join(text), encoding="utf-8")


def run_audit(cfg: dict) -> dict:
    """Inspect the Phase 1 main collection; persist a machine-checkable pass gate."""
    root = Path(cfg["paths"]["data_root"]).resolve()
    collection, output = root / "main", root / "audit"
    artifacts = Path(cfg["paths"]["artifacts_root"])
    if any(path.exists() for path in (root / "splits.csv", artifacts / "splits.sha256", artifacts / "FREEZE.lock")):
        raise ValueError("Splits/freeze already exist; do not rerun the data audit after sealing")
    source = json.loads((collection / "acquisition.json").read_text(encoding="utf-8"))
    manifest_path = collection / "manifest.csv"
    source_digest = sha256_of_file(manifest_path)
    if (source["status"] != "complete" or source["manifest_sha256"] != source_digest
        or source["specification"]["kind"] != "main"
        or source["specification"]["generator_name"] != cfg["dataset"]["generator_name"]):
        raise ValueError("Acquisition is incomplete, changed, or not the configured main collection")
    settings = cfg["audit"]
    threshold = settings["shortcut_max_accuracy"]
    if not .5 <= threshold < 1:
        raise ValueError("Shortcut threshold must be at least chance and less than one")
    digest = config_hash(cfg)
    specification = {"config_hash": digest, "source_manifest_sha256": source_digest}
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "audit_result.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8"))["specification"] != specification:
            raise ValueError("Audit folder belongs to different inputs/settings; inspect before rerunning")
    elif any(output.iterdir()):
        raise ValueError("Non-empty audit folder lacks provenance; choose/inspect the data root")
    _atomic_json(result_path, {"specification": specification, "status": "incomplete"})
    original = audit_table(pd.read_csv(manifest_path, dtype={"id": str}), collection,
                           phash_size=settings["phash_size"])
    # Resolve undersized-image eligibility before any split is frozen.
    eligible = original[["width", "height"]].min(axis=1) >= cfg["image_size"]
    if len(original[eligible & (original["label"] == 1)]) < cfg["dataset"]["minimum_ai_images"]:
        raise ValueError("Too few eligible AI images; consult the supervisor before finalizing data")
    before = balance_classes(original[eligible], cfg["seed"])
    excluded = original[~original["id"].isin(before["id"])].copy()
    excluded["reason"] = np.where(excluded[["width", "height"]].min(axis=1) < cfg["image_size"],
                                   "smaller_than_target", "excess_real_for_balance")
    for frame in (original, before, excluded):
        frame["source_config_hash"] = frame["config_hash"]
        frame["config_hash"] = digest
    original.to_csv(output / "original_audit.csv", index=False)
    excluded.to_csv(output / "excluded.csv", index=False)
    options = {"cv_folds": cfg["cv_folds"], "tree_depth": settings["probe_tree_depth"],
               "logistic_max_iter": settings["probe_logistic_max_iter"]}
    print("Running balanced metadata-only probes before remediation", flush=True)
    first = shortcut_probe(before, cfg["seed"], **options)
    active = before.copy()
    active["original_sha256"], active["original_phash"] = active["sha256"], active["phash"]
    active["rel_path"] = "main/" + active["rel_path"]
    target_quality = None
    remediation_error = None
    final = first
    if first["max_accuracy"] > threshold:
        real_quality = before.loc[before["label"] == 0, "jpeg_quality"].dropna()
        if real_quality.empty:
            remediation_error = "No real-image JPEG quality exists for the preplanned fix; consult supervisor"
        else:
            target_quality = int(round(float(real_quality.median())))
            print(f"Re-encoding BOTH classes identically at quality {target_quality}, 4:4:4", flush=True)
            normalized = []
            for row in active.to_dict("records"):
                payload = normalise_format((root / row["rel_path"]).read_bytes(), target_quality)
                saved = _save_original(output / "normalized", str(row["id"]), int(row["label"]),
                                       row["generator"], row["source_split"], payload, digest)
                description = describe_file(output / "normalized" / saved["rel_path"], phash_size=settings["phash_size"])
                normalized.append({**row, **saved, **description,
                                   "rel_path": "audit/normalized/" + saved["rel_path"]})
            active = pd.DataFrame(normalized)
            final = shortcut_probe(active, cfg["seed"], **options)
    active["group_id"] = duplicate_groups(active, settings["duplicate_hamming_distance"])
    active.to_csv(output / "active_manifest.csv", index=False)
    sizes = active.groupby("group_id").size()
    result = {"specification": specification, "status": "passed" if final["max_accuracy"] <= threshold else "blocked",
        "config_hash": digest, "threshold": threshold, "before_probe": first, "final_probe": final,
        "normalization_applied": target_quality is not None, "target_jpeg_quality": target_quality,
        "remediation_error": remediation_error,
        "active_manifest_sha256": sha256_of_file(output / "active_manifest.csv"),
        "original_rows": len(original), "eligible_rows": int(eligible.sum()), "excluded_rows": len(excluded),
        "exclusion_reasons": excluded["reason"].value_counts().to_dict(),
        "balanced_rows": len(active), "duplicate_groups": int((sizes > 1).sum()),
        "exact_duplicate_groups": int((active.groupby("sha256").size() > 1).sum()),
        "mixed_label_groups": int((active.groupby("group_id")["label"].nunique() > 1).sum())}
    _write_audit_report(original, before, active, result, Path(cfg["paths"]["reports_root"]), settings)
    _atomic_json(result_path, result)
    print("Audit result:", result["status"], "maximum probe accuracy:", final["max_accuracy"], flush=True)
    return result


def run_split_freeze(cfg: dict) -> dict:
    """Require the current passing audit and unchanged active image files."""
    root = Path(cfg["paths"]["data_root"]).resolve()
    artifacts = Path(cfg["paths"]["artifacts_root"])
    if (artifacts / "FREEZE.lock").exists():
        raise ValueError("Final evaluation freeze exists; Phase 2 cannot run")
    audit_dir = root / "audit"
    result = json.loads((audit_dir / "audit_result.json").read_text(encoding="utf-8"))
    if result["status"] != "passed" or result["config_hash"] != config_hash(cfg):
        raise ValueError("A passing audit for the current configuration is required before splits")
    manifest_path = audit_dir / "active_manifest.csv"
    if sha256_of_file(manifest_path) != result["active_manifest_sha256"]:
        raise ValueError("Audited manifest changed; do not create splits")
    active = pd.read_csv(manifest_path, dtype={"id": str, "group_id": str})
    for row in active.itertuples():
        path = (root / row.rel_path).resolve()
        if not path.is_relative_to(root) or sha256_of_file(path) != row.sha256:
            raise ValueError(f"Audited image changed or escaped its root: {row.id}")
    settings = cfg["splitting"]
    splits = make_group_splits(active, cfg["split"], cfg["seed"],
        tolerance=settings["proportion_tolerance"], time_limit=settings["solver_time_limit_seconds"])
    digest = freeze_splits(splits, root / "splits.csv", artifacts / "splits.sha256",
        {"config_hash": config_hash(cfg), "active_manifest_sha256": result["active_manifest_sha256"],
         "audit_result_sha256": sha256_of_file(audit_dir / "audit_result.json")})
    counts = validate_splits(splits, cfg["split"], settings["proportion_tolerance"])
    print("Split counts:", json.dumps(counts, sort_keys=True))
    print("Duplicate groups crossing splits: 0")
    print("Splits SHA-256:", digest)
    return {"counts": counts, "splits_sha256": digest, "crossing_groups": 0}
