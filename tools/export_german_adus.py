#!/usr/bin/env python3
"""Export the verified German ADU reference and bilingual ADU alignment."""

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
    BOUNDARY_FIELDS,
    REPO_ROOT,
    AuditError,
    CorpusAudit,
    build_audit,
    sha256_text,
    write_json,
)


DEFAULT_ADU_DIR = REPO_ROOT / "derived" / "adu" / "de" / "original_gold"
DEFAULT_BOUNDARY_PATH = REPO_ROOT / "derived" / "boundaries" / "de_adu_gold.tsv"
DEFAULT_ALIGNMENT_PATH = REPO_ROOT / "derived" / "alignments" / "adu_de_en.tsv"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "derived" / "adu" / "de" / "original_gold.manifest.json"
ALIGNMENT_FIELDS = [
    "doc_id",
    "adu_id",
    "adu_index",
    "de_start",
    "de_end",
    "en_start",
    "en_end",
    "de_text",
    "en_text",
]


def adu_file_text(document) -> str:
    return "\n".join(span.text for span in document.adus) + "\n"


def boundary_rows(audit: CorpusAudit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = f"arg-microtexts-corpus-tree@{audit.original_corpus_tree}"
    for document in audit.german_documents:
        for index, span in enumerate(document.adus):
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "language": "de",
                    "char_offset": span.start,
                    "token_index": "",
                    "adu_id": span.adu_id or "",
                    "edu_id": "",
                    "boundary_class": "document_start" if index == 0 else "adu",
                    "source": source,
                    "status": "gold",
                    "confidence": "",
                    "model": "",
                    "run_id": "",
                    "sameunit_affected": str(document.sameunit_affected).lower(),
                }
            )
    return rows


def alignment_rows(audit: CorpusAudit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    english_by_id = {document.doc_id: document for document in audit.english_documents}
    for german in audit.german_documents:
        english = english_by_id[german.doc_id]
        if len(german.adus) != len(english.original_adus):
            raise AuditError(f"Bilingual ADU count differs for {german.doc_id}")
        for index, (de_adu, en_adu) in enumerate(
            zip(german.adus, english.original_adus), start=1
        ):
            if de_adu.adu_id != en_adu.adu_id:
                raise AuditError(
                    f"Bilingual ADU ID differs for {german.doc_id} at position {index}: "
                    f"{de_adu.adu_id} != {en_adu.adu_id}"
                )
            rows.append(
                {
                    "doc_id": german.doc_id,
                    "adu_id": de_adu.adu_id or "",
                    "adu_index": index,
                    "de_start": de_adu.start,
                    "de_end": de_adu.end,
                    "en_start": en_adu.start,
                    "en_end": en_adu.end,
                    "de_text": de_adu.text,
                    "en_text": en_adu.text,
                }
            )
    return rows


def tsv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def expected_outputs(
    audit: CorpusAudit,
) -> tuple[dict[str, str], str, str, dict[str, object]]:
    files = {document.doc_id: adu_file_text(document) for document in audit.german_documents}
    boundaries = tsv_text(boundary_rows(audit), BOUNDARY_FIELDS)
    alignments = tsv_text(alignment_rows(audit), ALIGNMENT_FIELDS)
    manifest = {
        "schema_version": 1,
        "collection": "German original gold ADU reference",
        "language": "de",
        "status": "gold-adu-reference",
        "source": {
            "path": "corpus/de",
            "corpus_tree_sha": audit.original_corpus_tree,
        },
        "policy": {
            "raw_input": "Original corpus/de/*.txt files are canonical complete documents.",
            "unit_status": (
                "ADU boundaries are locked gold EDU boundaries; within-ADU German EDU "
                "boundaries remain unknown."
            ),
            "reconstruction": "Join stripped ADU lines with one ASCII space.",
            "bilingual_alignment": (
                "German and professional English translations share original ADU IDs and order."
            ),
        },
        "counts": {
            "documents": len(audit.german_documents),
            "adus": sum(len(document.adus) for document in audit.german_documents),
            "boundary_rows": len(boundary_rows(audit)),
            "alignment_rows": len(alignment_rows(audit)),
        },
        "boundary_inventory": {
            "path": str(DEFAULT_BOUNDARY_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_text(boundaries),
        },
        "bilingual_alignment": {
            "path": str(DEFAULT_ALIGNMENT_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_text(alignments),
        },
        "documents": [
            {
                "doc_id": document.doc_id,
                "raw_path": f"corpus/de/{document.doc_id}.txt",
                "adu_path": str(
                    (DEFAULT_ADU_DIR / f"{document.doc_id}.adus").relative_to(REPO_ROOT)
                ),
                "adu_count": len(document.adus),
                "raw_sha256": sha256_text(document.raw_text),
                "adu_file_sha256": sha256_text(files[document.doc_id]),
            }
            for document in audit.german_documents
        ],
    }
    return files, boundaries, alignments, manifest


def write_outputs(
    audit: CorpusAudit,
    adu_dir: Path,
    boundary_path: Path,
    alignment_path: Path,
    manifest_path: Path,
) -> None:
    files, boundaries, alignments, manifest = expected_outputs(audit)
    adu_dir.mkdir(parents=True, exist_ok=True)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    for doc_id, text in files.items():
        (adu_dir / f"{doc_id}.adus").write_text(text, encoding="utf-8")
    boundary_path.write_text(boundaries, encoding="utf-8")
    alignment_path.write_text(alignments, encoding="utf-8")
    write_json(manifest_path, manifest)


def check_outputs(
    audit: CorpusAudit,
    adu_dir: Path,
    boundary_path: Path,
    alignment_path: Path,
    manifest_path: Path,
) -> None:
    files, boundaries, alignments, manifest = expected_outputs(audit)
    for doc_id, expected in files.items():
        path = adu_dir / f"{doc_id}.adus"
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise AuditError(f"German ADU export differs or is missing: {path}")
    unexpected = {path.stem for path in adu_dir.glob("*.adus")} - set(files)
    if unexpected:
        raise AuditError(f"Unexpected German ADU exports: {sorted(unexpected)}")
    if not boundary_path.exists() or boundary_path.read_text(encoding="utf-8") != boundaries:
        raise AuditError(f"German boundary inventory differs or is missing: {boundary_path}")
    if not alignment_path.exists() or alignment_path.read_text(encoding="utf-8") != alignments:
        raise AuditError(f"Bilingual ADU alignment differs or is missing: {alignment_path}")
    if not manifest_path.exists():
        raise AuditError(f"German ADU manifest is missing: {manifest_path}")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise AuditError(f"German ADU manifest differs: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check committed outputs without writing.")
    parser.add_argument("--adu-dir", type=Path, default=DEFAULT_ADU_DIR)
    parser.add_argument("--boundary-path", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--alignment-path", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = build_audit()
        if args.check:
            check_outputs(
                audit,
                args.adu_dir,
                args.boundary_path,
                args.alignment_path,
                args.manifest_path,
            )
            print("GERMAN ADU EXPORT CHECK PASSED")
        else:
            write_outputs(
                audit,
                args.adu_dir,
                args.boundary_path,
                args.alignment_path,
                args.manifest_path,
            )
            print(f"WROTE 576 German ADUs in {len(audit.german_documents)} documents")
    except (AuditError, OSError, ValueError) as exc:
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
