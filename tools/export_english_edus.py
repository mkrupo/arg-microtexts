#!/usr/bin/env python3
"""Export the verified English multilayer EDU segmentation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import (
    REPO_ROOT,
    AuditError,
    CorpusAudit,
    build_audit,
    sha256_text,
    write_json,
)


DEFAULT_EDU_DIR = REPO_ROOT / "derived" / "edu" / "en" / "multilayer_gold"
DEFAULT_BOUNDARY_PATH = REPO_ROOT / "derived" / "boundaries" / "en_multilayer_gold.tsv"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "derived" / "edu" / "en" / "multilayer_gold.manifest.json"
BOUNDARY_FIELDS = [
    "doc_id",
    "language",
    "char_offset",
    "token_index",
    "adu_id",
    "edu_id",
    "boundary_class",
    "source",
    "status",
    "confidence",
    "model",
    "run_id",
    "sameunit_affected",
]


def edu_file_text(document) -> str:
    return "\n".join(span.text for span in document.edus) + "\n"


def boundary_rows(audit: CorpusAudit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = f"arg-microtexts-multilayer@{audit.source_commit}"
    for document in audit.english_documents:
        seen_adus: set[str] = set()
        for index, span in enumerate(document.edus):
            adu_id = span.adu_id or ""
            if index == 0:
                boundary_class = "document_start"
            elif adu_id not in seen_adus:
                boundary_class = "adu"
            else:
                boundary_class = "internal_edu"
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "language": "en",
                    "char_offset": span.start,
                    "token_index": "",
                    "adu_id": adu_id,
                    "edu_id": span.unit_id,
                    "boundary_class": boundary_class,
                    "source": source,
                    "status": "gold",
                    "confidence": "",
                    "model": "",
                    "run_id": "",
                    "sameunit_affected": str(document.sameunit_affected).lower(),
                }
            )
            seen_adus.add(adu_id)
    return rows


def boundary_tsv(audit: CorpusAudit) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BOUNDARY_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(boundary_rows(audit))
    return stream.getvalue()


def export_manifest(audit: CorpusAudit, files: dict[str, str], boundary_text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "collection": "English multilayer gold EDU segmentation",
        "language": "en",
        "status": "gold",
        "source": {
            "repository": "https://github.com/peldszus/arg-microtexts-multilayer.git",
            "commit": audit.source_commit,
            "layer": "corpus/rst",
            "adu_mapping_layer": "corpus/arg",
        },
        "policy": {
            "primary_reference": "main 680-EDU common multilayer segmentation",
            "sameunit_alternatives": (
                "Nine affected documents are flagged; no-Same-Unit RST trees are not the "
                "primary segmentation."
            ),
            "reconstruction": "Join stripped EDU lines with one ASCII space.",
        },
        "known_source_variants": audit.manifest()["known_source_variants"],
        "counts": audit.totals,
        "boundary_inventory": {
            "path": str(DEFAULT_BOUNDARY_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_text(boundary_text),
            "rows": len(boundary_rows(audit)),
        },
        "documents": [
            {
                "doc_id": document.doc_id,
                "path": str((DEFAULT_EDU_DIR / f"{document.doc_id}.edus").relative_to(REPO_ROOT)),
                "edu_count": len(document.edus),
                "adu_count": len(document.original_adus),
                "sameunit_affected": document.sameunit_affected,
                "raw_sha256": sha256_text(document.raw_text),
                "edu_file_sha256": sha256_text(files[document.doc_id]),
            }
            for document in audit.english_documents
        ],
    }


def expected_outputs(audit: CorpusAudit) -> tuple[dict[str, str], str, dict[str, object]]:
    files = {document.doc_id: edu_file_text(document) for document in audit.english_documents}
    boundaries = boundary_tsv(audit)
    return files, boundaries, export_manifest(audit, files, boundaries)


def check_outputs(audit: CorpusAudit, edu_dir: Path, boundary_path: Path, manifest_path: Path) -> None:
    files, boundaries, manifest = expected_outputs(audit)
    for doc_id, expected in files.items():
        path = edu_dir / f"{doc_id}.edus"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise AuditError(f"English EDU export differs or is missing: {path}")
    unexpected = {path.stem for path in edu_dir.glob("*.edus")} - set(files)
    if unexpected:
        raise AuditError(f"Unexpected English EDU exports: {sorted(unexpected)}")
    if not boundary_path.exists() or boundary_path.read_text(encoding="utf-8") != boundaries:
        raise AuditError(f"Boundary inventory differs or is missing: {boundary_path}")
    if not manifest_path.exists():
        raise AuditError(f"Manifest is missing: {manifest_path}")
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != manifest:
        raise AuditError(f"Manifest differs: {manifest_path}")


def write_outputs(audit: CorpusAudit, edu_dir: Path, boundary_path: Path, manifest_path: Path) -> None:
    files, boundaries, manifest = expected_outputs(audit)
    edu_dir.mkdir(parents=True, exist_ok=True)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    for doc_id, text in files.items():
        (edu_dir / f"{doc_id}.edus").write_text(text, encoding="utf-8")
    boundary_path.write_text(boundaries, encoding="utf-8")
    write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check committed outputs without writing.")
    parser.add_argument("--edu-dir", type=Path, default=DEFAULT_EDU_DIR)
    parser.add_argument("--boundary-path", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = build_audit()
        if args.check:
            check_outputs(audit, args.edu_dir, args.boundary_path, args.manifest_path)
            print("ENGLISH EXPORT CHECK PASSED")
        else:
            write_outputs(audit, args.edu_dir, args.boundary_path, args.manifest_path)
            print(
                f"WROTE {audit.totals['english_edus']} EDUs in "
                f"{audit.totals['documents']} documents"
            )
    except (AuditError, OSError, ValueError) as exc:
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
