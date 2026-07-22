#!/usr/bin/env python3
"""Run EduSeg independently on each German ADU as a context ablation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import BOUNDARY_FIELDS, REPO_ROOT, AuditError, build_audit, write_json
from tools.run_eduseg_de import (
    DocumentPrediction,
    git_head,
    load_config,
    output_hashes,
    predict_documents,
    prepare_output,
    repository_path,
    segment_text,
    select_device,
    sha256_file,
    verify_model,
)


DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "eduseg_de_adu_context_v1.toml"
DEFAULT_OUTPUT = REPO_ROOT / "work" / "runs" / "eduseg_de_adu_context_v1"
ADU_SCORE_FIELDS = [
    "doc_id",
    "adu_id",
    "adu_token_index",
    "char_offset",
    "char_end",
    "token_text",
    "boundary_probability",
    "predicted_boundary",
]
COMPARISON_FIELDS = [
    "doc_id",
    "adu_id",
    "char_offset",
    "document_predicted",
    "adu_context_predicted",
    "document_probability",
    "adu_context_probability",
    "left_context",
    "right_context",
]


@dataclass(frozen=True)
class ADUInput:
    doc_id: str
    raw_text: str
    parent_doc_id: str
    adu_id: str
    document_start: int


@dataclass(frozen=True)
class MergedPrediction:
    doc_id: str
    internal_starts: frozenset[int]
    score_rows: tuple[dict[str, object], ...]
    invalid_offsets: tuple[int, ...]
    max_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_adu_inputs(audit) -> list[ADUInput]:
    return [
        ADUInput(
            doc_id=f"{document.doc_id}/{adu.adu_id}",
            raw_text=adu.text,
            parent_doc_id=document.doc_id,
            adu_id=adu.adu_id or "",
            document_start=adu.start,
        )
        for document in audit.german_documents
        for adu in document.adus
    ]


def merge_predictions(
    audit, inputs: list[ADUInput], predictions: list[DocumentPrediction]
) -> list[MergedPrediction]:
    if len(inputs) != len(predictions):
        raise AuditError("ADU input and prediction counts differ")
    grouped_starts: dict[str, set[int]] = {}
    grouped_scores: dict[str, list[dict[str, object]]] = {}
    grouped_invalid: dict[str, list[int]] = {}
    grouped_tokens: dict[str, list[int]] = {}
    for item, prediction in zip(inputs, predictions):
        if prediction.doc_id != item.doc_id:
            raise AuditError(f"ADU prediction order differs: {item.doc_id} != {prediction.doc_id}")
        starts = grouped_starts.setdefault(item.parent_doc_id, set())
        starts.update(
            item.document_start + start
            for start in prediction.predicted_starts
            if start != 0
        )
        score_rows = grouped_scores.setdefault(item.parent_doc_id, [])
        for score in prediction.scores:
            score_rows.append(
                {
                    "doc_id": item.parent_doc_id,
                    "adu_id": item.adu_id,
                    "adu_token_index": score.model_token_index,
                    "char_offset": item.document_start + score.start,
                    "char_end": item.document_start + score.end,
                    "token_text": score.token_text,
                    "boundary_probability": f"{score.probability:.8f}",
                    "predicted_boundary": str(score.predicted).lower(),
                }
            )
        grouped_invalid.setdefault(item.parent_doc_id, []).extend(
            item.document_start + offset for offset in prediction.invalid_boundary_offsets
        )
        grouped_tokens.setdefault(item.parent_doc_id, []).append(prediction.token_count)

    return [
        MergedPrediction(
            doc_id=document.doc_id,
            internal_starts=frozenset(grouped_starts.get(document.doc_id, set())),
            score_rows=tuple(grouped_scores.get(document.doc_id, [])),
            invalid_offsets=tuple(sorted(set(grouped_invalid.get(document.doc_id, [])))),
            max_tokens=max(grouped_tokens[document.doc_id]),
        )
        for document in audit.german_documents
    ]


def strongest_scores(rows: list[dict[str, str]]) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["doc_id"], int(row["char_offset"]))
        scores[key] = max(scores.get(key, 0.0), float(row["boundary_probability"]))
    return scores


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def boundary_rows(audit, merged: list[MergedPrediction], config) -> list[dict[str, object]]:
    predictions = {prediction.doc_id: prediction for prediction in merged}
    rows: list[dict[str, object]] = []
    for document in audit.german_documents:
        prediction = predictions[document.doc_id]
        score_by_start = strongest_scores(list(prediction.score_rows))
        adu_by_start = {adu.start: adu for adu in document.adus}
        starts = sorted(set(adu_by_start) | set(prediction.internal_starts))
        for index, start in enumerate(starts, start=1):
            if start in adu_by_start:
                adu = adu_by_start[start]
                boundary_class = "document_start" if start == 0 else "adu"
                source = "structural_constraint"
                status = "forced"
                confidence = ""
                model = ""
            else:
                adu = next(adu for adu in document.adus if adu.start < start < adu.end)
                boundary_class = "internal_edu"
                source = config["model"]["name"]
                status = "automatic"
                confidence = f"{score_by_start[(document.doc_id, start)]:.8f}"
                model = f"{config['model']['name']}@{config['model']['revision']}"
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "language": "de",
                    "char_offset": start,
                    "token_index": "",
                    "adu_id": adu.adu_id or "",
                    "edu_id": f"e{index}",
                    "boundary_class": boundary_class,
                    "source": source,
                    "status": status,
                    "confidence": confidence,
                    "model": model,
                    "run_id": config["run_id"],
                    "sameunit_affected": str(document.sameunit_affected).lower(),
                }
            )
    return rows


def comparison_rows(audit, merged: list[MergedPrediction], config) -> list[dict[str, object]]:
    document_boundaries = read_tsv(REPO_ROOT / str(config["comparison_boundaries"]))
    document_starts = {
        (row["doc_id"], int(row["char_offset"]))
        for row in document_boundaries
        if row["boundary_class"] == "internal_edu"
    }
    document_scores = strongest_scores(
        read_tsv(REPO_ROOT / "derived" / "scores" / "de_eduseg_de_document_v1.tsv")
    )
    adu_starts = {
        (prediction.doc_id, start)
        for prediction in merged
        for start in prediction.internal_starts
    }
    adu_scores = strongest_scores(
        [row for prediction in merged for row in prediction.score_rows]
    )
    documents = {document.doc_id: document for document in audit.german_documents}
    rows: list[dict[str, object]] = []
    for doc_id, start in sorted(document_starts | adu_starts):
        document = documents[doc_id]
        adu = next(adu for adu in document.adus if adu.start < start < adu.end)
        rows.append(
            {
                "doc_id": doc_id,
                "adu_id": adu.adu_id or "",
                "char_offset": start,
                "document_predicted": str((doc_id, start) in document_starts).lower(),
                "adu_context_predicted": str((doc_id, start) in adu_starts).lower(),
                "document_probability": f"{document_scores[(doc_id, start)]:.8f}",
                "adu_context_probability": f"{adu_scores[(doc_id, start)]:.8f}",
                "left_context": document.raw_text[max(0, start - 100) : start].strip(),
                "right_context": document.raw_text[start : start + 100].strip(),
            }
        )
    return rows


def make_summary(
    merged: list[MergedPrediction],
    comparisons: list[dict[str, object]],
    inference_adus: int,
) -> dict[str, object]:
    both = sum(
        row["document_predicted"] == "true" and row["adu_context_predicted"] == "true"
        for row in comparisons
    )
    document_only = sum(
        row["document_predicted"] == "true" and row["adu_context_predicted"] == "false"
        for row in comparisons
    )
    adu_only = sum(
        row["document_predicted"] == "false" and row["adu_context_predicted"] == "true"
        for row in comparisons
    )
    union = both + document_only + adu_only
    return {
        "documents": len(merged),
        "inference_adus": inference_adus,
        "document_context_internal_proposals": both + document_only,
        "adu_context_internal_proposals": both + adu_only,
        "shared_internal_proposals": both,
        "document_context_only": document_only,
        "adu_context_only": adu_only,
        "internal_boundary_jaccard": both / union if union else 1.0,
        "internal_boundary_f1_agreement": (
            2 * both / (2 * both + document_only + adu_only)
            if both or document_only or adu_only
            else 1.0
        ),
        "max_adu_model_tokens": max(prediction.max_tokens for prediction in merged),
        "invalid_subword_boundary_predictions": sum(
            len(prediction.invalid_offsets) for prediction in merged
        ),
        "interpretation": (
            "Agreement between inference contexts, not accuracy against German EDU gold."
        ),
    }


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        config = load_config(args.config)
        model_hashes = verify_model(args.model_dir, config)
        audit = build_audit()
        inputs = make_adu_inputs(audit)
        try:
            import torch
            import transformers
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise AuditError("Install torch and transformers to run EduSeg inference") from exc
        device = select_device(args.device, torch)
        batch_size = args.batch_size or int(config["inference"]["batch_size"])
        if batch_size <= 0:
            raise AuditError(f"Batch size must be positive, found {batch_size}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if not tokenizer.is_fast:
            raise AuditError("EduSeg inference requires a fast tokenizer for character offsets")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir, local_files_only=True
        ).to(device)
        boundary_label_id = int(config["model"]["boundary_label_id"])
        inside_label_id = int(config["model"]["inside_label_id"])
        if {inside_label_id, boundary_label_id} != set(range(model.config.num_labels)):
            raise AuditError("Model/config label mapping differs")
        predictions = predict_documents(
            documents=inputs,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            boundary_label_id=boundary_label_id,
            max_length=int(config["model"]["max_length"]),
            batch_size=batch_size,
            require_text_boundary=bool(config["inference"]["require_text_boundary"]),
        )
        merged = merge_predictions(audit, inputs, predictions)
        boundaries = boundary_rows(audit, merged, config)
        comparisons = comparison_rows(audit, merged, config)
        summary = make_summary(merged, comparisons, len(inputs))
        prepare_output(args.output_dir, args.overwrite)
        edu_dir = args.output_dir / "edus"
        edu_dir.mkdir()
        merged_by_doc = {prediction.doc_id: prediction for prediction in merged}
        for document in audit.german_documents:
            starts = {adu.start for adu in document.adus}
            starts |= set(merged_by_doc[document.doc_id].internal_starts)
            edus = segment_text(document.raw_text, starts)
            (edu_dir / f"{document.doc_id}.edus").write_text(
                "\n".join(edus) + "\n", encoding="utf-8"
            )
        all_scores = [row for prediction in merged for row in prediction.score_rows]
        write_tsv(args.output_dir / "boundaries.tsv", boundaries, BOUNDARY_FIELDS)
        write_tsv(args.output_dir / "boundary_scores.tsv", all_scores, ADU_SCORE_FIELDS)
        write_tsv(args.output_dir / "context_comparison.tsv", comparisons, COMPARISON_FIELDS)
        write_json(args.output_dir / "summary.json", summary)
        manifest = {
            "schema_version": 1,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "project_commit": git_head(),
            "config_path": repository_path(args.config),
            "config_sha256": sha256_file(args.config),
            "source_audit_sha256": sha256_file(
                REPO_ROOT / "derived" / "source_audit.manifest.json"
            ),
            "model": {**config["model"], "verified_sha256": model_hashes},
            "inference": {
                **config["inference"],
                "unit": "individual_adu",
                "device": str(device),
                "batch_size": batch_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "comparison": {
                "run_id": config["comparison_run"],
                "boundary_path": config["comparison_boundaries"],
                "boundary_sha256": sha256_file(
                    REPO_ROOT / str(config["comparison_boundaries"])
                ),
            },
            "outputs": {
                "edu_files": len(list(edu_dir.glob("*.edus"))),
                "boundary_rows": len(boundaries),
                "score_rows": len(all_scores),
                "comparison_rows": len(comparisons),
                "files": output_hashes(args.output_dir),
            },
            "summary": summary,
        }
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (AuditError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"EDUSEG ADU ABLATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
