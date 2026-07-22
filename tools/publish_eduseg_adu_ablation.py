#!/usr/bin/env python3
"""Verify and publish the EduSeg per-ADU context ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import REPO_ROOT, AuditError, build_audit
from tools.publish_eduseg_run import json_text, read_tsv
from tools.run_eduseg_de import sha256_file


RUN_ID = "eduseg_de_adu_context_v1"
DEFAULT_RUN_DIR = REPO_ROOT / "work" / "runs" / RUN_ID
EDU_DIR = REPO_ROOT / "derived" / "edu" / "de" / "automatic" / RUN_ID / "adu_constrained"
BOUNDARY_PATH = REPO_ROOT / "derived" / "boundaries" / f"de_{RUN_ID}.tsv"
SCORE_PATH = REPO_ROOT / "derived" / "scores" / f"de_{RUN_ID}.tsv"
RESULT_DIR = REPO_ROOT / "experiments" / "results" / RUN_ID
COMPARISON_PATH = RESULT_DIR / "context_comparison.tsv"
SUMMARY_PATH = RESULT_DIR / "summary.json"
RUN_MANIFEST_PATH = RESULT_DIR / "run_manifest.json"
MANIFEST_PATH = RESULT_DIR / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--check", action="store_true", help="Check published files only.")
    return parser.parse_args()


def verify_run(run_dir: Path, manifest: dict[str, object]) -> None:
    if manifest.get("run_id") != RUN_ID:
        raise AuditError(f"Expected run_id={RUN_ID}, found {manifest.get('run_id')}")
    config_path = REPO_ROOT / str(manifest["config_path"])
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise AuditError(f"Run configuration hash differs: {config_path}")
    expected = {"manifest.json"}
    for item in manifest["outputs"]["files"]:
        relative = str(item["path"])
        expected.add(relative)
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise AuditError(f"Run artifact path escapes run directory: {relative}") from exc
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"Run artifact differs or is missing: {path}")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise AuditError(
            f"Run artifact set differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def verify_edus(directory: Path) -> None:
    audit = build_audit()
    documents = {document.doc_id: document for document in audit.german_documents}
    paths = {path.stem: path for path in directory.glob("*.edus")}
    if set(paths) != set(documents):
        raise AuditError(f"Ablation document set differs in {directory}")
    for doc_id, path in paths.items():
        reconstructed = " ".join(path.read_text(encoding="utf-8").splitlines())
        if reconstructed != documents[doc_id].raw_text:
            raise AuditError(f"Ablation EDUs do not reconstruct raw text: {path}")


def expected_outputs(run_dir: Path) -> dict[Path, str]:
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    verify_run(run_dir, run_manifest)
    verify_edus(run_dir / "edus")
    outputs: dict[Path, str] = {}
    for source in sorted((run_dir / "edus").glob("*.edus")):
        outputs[EDU_DIR / source.name] = source.read_text(encoding="utf-8")
    for source, target in (
        (run_dir / "boundaries.tsv", BOUNDARY_PATH),
        (run_dir / "boundary_scores.tsv", SCORE_PATH),
        (run_dir / "context_comparison.tsv", COMPARISON_PATH),
    ):
        outputs[target] = source.read_text(encoding="utf-8")
    outputs[SUMMARY_PATH] = json_text(run_manifest["summary"])
    outputs[RUN_MANIFEST_PATH] = json_text(run_manifest)
    files = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for path, text in sorted(outputs.items(), key=lambda item: str(item[0]))
    ]
    public_manifest = {
        "schema_version": 1,
        "collection": "German EduSeg per-ADU context ablation",
        "language": "de",
        "status": "automatic-ablation-not-gold",
        "run_id": RUN_ID,
        "source_run": {
            "project_commit": run_manifest["project_commit"],
            "model": run_manifest["model"],
            "inference": run_manifest["inference"],
            "comparison": run_manifest["comparison"],
            "completed_at": run_manifest["completed_at"],
            "run_manifest_path": RUN_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        },
        "policy": {
            "role": "Context-sensitivity ablation; not the primary German segmentation.",
            "adu_starts": "Inserted structurally because every ADU is a separate input.",
            "comparison": "Only strictly within-ADU starts are compared at character offsets.",
            "confidence": "Uncalibrated softmax probability for the boundary label.",
        },
        "counts": {
            "documents": run_manifest["summary"]["documents"],
            "inference_adus": run_manifest["summary"]["inference_adus"],
            "edus": run_manifest["outputs"]["boundary_rows"],
            "model_token_scores": run_manifest["outputs"]["score_rows"],
            "comparison_rows": run_manifest["outputs"]["comparison_rows"],
        },
        "summary": run_manifest["summary"],
        "publication_tool": {
            "path": "tools/publish_eduseg_adu_ablation.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "files": files,
    }
    outputs[MANIFEST_PATH] = json_text(public_manifest)
    return outputs


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def check_published() -> None:
    if not MANIFEST_PATH.is_file():
        raise AuditError(f"Ablation manifest is missing: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID:
        raise AuditError(f"Published ablation run ID differs: {manifest.get('run_id')}")
    if manifest["publication_tool"]["sha256"] != sha256_file(Path(__file__)):
        raise AuditError("Ablation publication tool hash differs")
    for item in manifest["files"]:
        path = (REPO_ROOT / item["path"]).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise AuditError(f"Published path escapes repository: {path}") from exc
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"Published ablation artifact differs or is missing: {path}")
    verify_edus(EDU_DIR)
    if len(read_tsv(COMPARISON_PATH)) != manifest["counts"]["comparison_rows"]:
        raise AuditError("Published ablation comparison count differs")


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            check_published()
            print("EDUSEG ADU ABLATION PUBLICATION CHECK PASSED")
        else:
            outputs = expected_outputs(args.run_dir)
            write_outputs(outputs)
            check_published()
            print("PUBLISHED 112 German EduSeg per-ADU ablation files")
    except (AuditError, KeyError, OSError, ValueError) as exc:
        print(f"ABLATION PUBLICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
