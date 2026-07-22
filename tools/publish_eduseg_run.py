#!/usr/bin/env python3
"""Verify and publish the frozen complete-document EduSeg proposal layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import REPO_ROOT, AuditError, build_audit
from tools.run_eduseg_de import sha256_file


RUN_ID = "eduseg_de_document_v1"
DEFAULT_RUN_DIR = REPO_ROOT / "work" / "runs" / RUN_ID
EDU_ROOT = REPO_ROOT / "derived" / "edu" / "de" / "automatic" / RUN_ID
RAW_EDU_DIR = EDU_ROOT / "raw"
CONSTRAINED_EDU_DIR = EDU_ROOT / "adu_constrained"
RAW_BOUNDARY_PATH = REPO_ROOT / "derived" / "boundaries" / f"de_{RUN_ID}_raw.tsv"
CONSTRAINED_BOUNDARY_PATH = (
    REPO_ROOT / "derived" / "boundaries" / f"de_{RUN_ID}_adu_constrained.tsv"
)
SCORE_PATH = REPO_ROOT / "derived" / "scores" / f"de_{RUN_ID}.tsv"
RESULT_DIR = REPO_ROOT / "experiments" / "results" / RUN_ID
SUMMARY_PATH = RESULT_DIR / "summary.json"
RUN_MANIFEST_PATH = RESULT_DIR / "run_manifest.json"
PROPOSAL_PATH = RESULT_DIR / "internal_proposals.tsv"
MISSED_ADU_PATH = RESULT_DIR / "missed_adu_boundaries.tsv"
MANIFEST_PATH = RESULT_DIR / "manifest.json"

PROPOSAL_FIELDS = [
    "doc_id",
    "adu_id",
    "char_offset",
    "boundary_probability",
    "after_terminal_punctuation",
    "left_candidate_segment",
    "right_candidate_segment",
    "adu_text",
]
MISSED_FIELDS = [
    "doc_id",
    "adu_id",
    "char_offset",
    "boundary_probability",
    "left_context",
    "adu_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--check", action="store_true", help="Check published files only.")
    return parser.parse_args()


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def tsv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def verify_run_files(run_dir: Path, manifest: dict[str, object]) -> None:
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


def verify_segmentations(run_dir: Path, audit) -> None:
    documents = {document.doc_id: document for document in audit.german_documents}
    expected_ids = set(documents)
    for directory_name in ("raw_edus", "constrained_edus"):
        directory = run_dir / directory_name
        paths = {path.stem: path for path in directory.glob("*.edus")}
        if set(paths) != expected_ids:
            raise AuditError(f"Document set differs in {directory}")
        for doc_id, path in paths.items():
            edus = path.read_text(encoding="utf-8").splitlines()
            if not edus or any(not edu for edu in edus):
                raise AuditError(f"Empty EDU in {path}")
            if " ".join(edus) != documents[doc_id].raw_text:
                raise AuditError(f"EDUs do not reconstruct raw text: {path}")


def internal_proposal_rows(run_dir: Path, audit) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    boundaries = read_tsv(run_dir / "constrained_boundaries.tsv")
    by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in boundaries:
        by_document[row["doc_id"]].append(row)

    rows: list[dict[str, object]] = []
    for doc_id, doc_boundaries in sorted(by_document.items()):
        document = documents[doc_id]
        ordered = sorted(doc_boundaries, key=lambda row: int(row["char_offset"]))
        starts = [int(row["char_offset"]) for row in ordered]
        for index, row in enumerate(ordered):
            if row["boundary_class"] != "internal_edu":
                continue
            start = starts[index]
            previous = starts[index - 1]
            following = starts[index + 1] if index + 1 < len(starts) else len(document.raw_text)
            adu = next(adu for adu in document.adus if adu.adu_id == row["adu_id"])
            rows.append(
                {
                    "doc_id": doc_id,
                    "adu_id": row["adu_id"],
                    "char_offset": start,
                    "boundary_probability": row["confidence"],
                    "after_terminal_punctuation": str(
                        document.raw_text[:start]
                        .rstrip()
                        .endswith((".", "!", "?", "…"))
                    ).lower(),
                    "left_candidate_segment": document.raw_text[previous:start].strip(),
                    "right_candidate_segment": document.raw_text[start:following].strip(),
                    "adu_text": adu.text,
                }
            )
    return rows


def missed_adu_rows(
    run_dir: Path, audit, run_manifest: dict[str, object]
) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    scores: dict[tuple[str, int], float] = {}
    for row in read_tsv(run_dir / "boundary_scores.tsv"):
        key = (row["doc_id"], int(row["char_offset"]))
        scores[key] = max(scores.get(key, 0.0), float(row["boundary_probability"]))
    rows: list[dict[str, object]] = []
    for missing in run_manifest["summary"]["missing_adu_boundaries"]:
        doc_id = str(missing["doc_id"])
        start = int(missing["char_offset"])
        text = documents[doc_id].raw_text
        rows.append(
            {
                "doc_id": doc_id,
                "adu_id": missing["adu_id"],
                "char_offset": start,
                "boundary_probability": f"{scores[(doc_id, start)]:.8f}",
                "left_context": text[max(0, start - 120) : start].strip(),
                "adu_text": missing["text"],
            }
        )
    return rows


def expected_outputs(run_dir: Path) -> dict[Path, str]:
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    verify_run_files(run_dir, run_manifest)
    audit = build_audit()
    verify_segmentations(run_dir, audit)
    outputs: dict[Path, str] = {}
    for source_dir, target_dir in (
        (run_dir / "raw_edus", RAW_EDU_DIR),
        (run_dir / "constrained_edus", CONSTRAINED_EDU_DIR),
    ):
        for source in sorted(source_dir.glob("*.edus")):
            outputs[target_dir / source.name] = source.read_text(encoding="utf-8")
    for source, target in (
        (run_dir / "raw_boundaries.tsv", RAW_BOUNDARY_PATH),
        (run_dir / "constrained_boundaries.tsv", CONSTRAINED_BOUNDARY_PATH),
        (run_dir / "boundary_scores.tsv", SCORE_PATH),
    ):
        outputs[target] = source.read_text(encoding="utf-8")
    proposals = internal_proposal_rows(run_dir, audit)
    missed = missed_adu_rows(run_dir, audit, run_manifest)
    outputs[PROPOSAL_PATH] = tsv_text(proposals, PROPOSAL_FIELDS)
    outputs[MISSED_ADU_PATH] = tsv_text(missed, MISSED_FIELDS)
    outputs[SUMMARY_PATH] = json_text(run_manifest["summary"])
    outputs[RUN_MANIFEST_PATH] = json_text(run_manifest)

    hashed_files = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for path, text in sorted(outputs.items(), key=lambda item: str(item[0]))
    ]
    public_manifest = {
        "schema_version": 1,
        "collection": "German EduSeg complete-document automatic proposals",
        "language": "de",
        "status": "automatic-proposal-not-gold",
        "run_id": RUN_ID,
        "source_run": {
            "project_commit": run_manifest["project_commit"],
            "model": run_manifest["model"],
            "inference": run_manifest["inference"],
            "completed_at": run_manifest["completed_at"],
            "run_manifest_path": RUN_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        },
        "policy": {
            "raw": "Unmodified valid textual boundary starts predicted by eduseg_de.",
            "adu_constrained": "Union of raw predictions and locked gold ADU starts.",
            "confidence": "Uncalibrated softmax probability for the boundary label.",
            "use": "Candidate German EDU segmentation for review; not human gold.",
        },
        "counts": {
            "documents": run_manifest["summary"]["documents"],
            "raw_edus": run_manifest["outputs"]["raw_boundary_rows"],
            "adu_constrained_edus": run_manifest["outputs"]["constrained_boundary_rows"],
            "model_token_scores": run_manifest["outputs"]["score_rows"],
        },
        "summary": run_manifest["summary"],
        "review_tables": {
            "internal_proposals": len(proposals),
            "missed_adu_boundaries": len(missed),
        },
        "publication_tool": {
            "path": "tools/publish_eduseg_run.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "files": hashed_files,
    }
    outputs[MANIFEST_PATH] = json_text(public_manifest)
    return outputs


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def check_published() -> None:
    if not MANIFEST_PATH.is_file():
        raise AuditError(f"Published manifest is missing: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID:
        raise AuditError(f"Published run ID differs: {manifest.get('run_id')}")
    if manifest["publication_tool"]["sha256"] != sha256_file(Path(__file__)):
        raise AuditError("Publication tool hash differs from the published manifest")
    for item in manifest["files"]:
        path = REPO_ROOT / item["path"]
        try:
            path.resolve().relative_to(REPO_ROOT)
        except ValueError as exc:
            raise AuditError(f"Published path escapes repository: {path}") from exc
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"Published artifact differs or is missing: {path}")

    audit = build_audit()
    documents = {document.doc_id: document for document in audit.german_documents}
    expected_ids = set(documents)
    for directory in (RAW_EDU_DIR, CONSTRAINED_EDU_DIR):
        paths = {path.stem: path for path in directory.glob("*.edus")}
        actual_ids = set(paths)
        if actual_ids != expected_ids:
            raise AuditError(f"Published document set differs in {directory}")
        for doc_id, path in paths.items():
            reconstructed = " ".join(path.read_text(encoding="utf-8").splitlines())
            if reconstructed != documents[doc_id].raw_text:
                raise AuditError(f"Published EDUs do not reconstruct raw text: {path}")
    if len(read_tsv(PROPOSAL_PATH)) != manifest["review_tables"]["internal_proposals"]:
        raise AuditError("Published internal-proposal count differs")
    if len(read_tsv(MISSED_ADU_PATH)) != manifest["review_tables"]["missed_adu_boundaries"]:
        raise AuditError("Published missed-ADU count differs")


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            check_published()
            print("EDUSEG PUBLICATION CHECK PASSED")
        else:
            outputs = expected_outputs(args.run_dir)
            write_outputs(outputs)
            check_published()
            print("PUBLISHED 112 raw and 112 ADU-constrained German EduSeg files")
    except (AuditError, KeyError, OSError, ValueError) as exc:
        print(f"PUBLICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
