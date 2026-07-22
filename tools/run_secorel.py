#!/usr/bin/env python3
"""Run SeCoRel on a frozen DISRPT-like token stream with character offsets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import BOUNDARY_FIELDS, REPO_ROOT, AuditError, build_audit, write_json
from tools.run_eduseg_de import (
    git_head,
    load_config,
    output_hashes,
    prepare_output,
    repository_path,
    segment_text,
    select_device,
    sha256_file,
    verify_model,
)


DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "secorel_disrpt_sentence_chunks_v1.toml"
DEFAULT_OUTPUT = REPO_ROOT / "work" / "runs" / "secorel_disrpt_sentence_chunks_v1"
TOKEN_RE = re.compile(
    r"https?://\S+|"
    r"[\w.+-]+@[\w.-]+\.\w+|"
    r"\.{3}|"
    r"--+|"
    r"\.\w+|"
    r"[A-Za-zÄÖÜäöüß]\.s|"
    r"(?:[A-Za-zÄÖÜäöüß]\.)+|"
    r"(?:bzw|etc|vgl|ca|Nr|Dr|Prof|bspw|evtl|ggf)\.|"
    r"\d+\.|"
    r"[’']s|"
    r"\w+-(?!\w)|"
    r"\w+(?:[.,:/—-]\w+)*|"
    r"[^\w\s]",
    re.UNICODE,
)
SENTENCE_END = {".", "!", "?", "؟", "۔", "።", "။", "፨", "。", "！", "？", "৷", "॥", "፣", "…", "ฯ"}
SCORE_FIELDS = [
    "doc_id",
    "token_index",
    "char_offset",
    "char_end",
    "token_text",
    "boundary_probability",
    "predicted_boundary",
    "gold_adu_start",
]
COMPARISON_FIELDS = [
    "doc_id",
    "adu_id",
    "char_offset",
    "eduseg_de_predicted",
    "secorel_predicted",
    "eduseg_de_probability",
    "secorel_probability",
    "left_context",
    "right_context",
]


@dataclass(frozen=True)
class TokenSpan:
    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenScore:
    token: TokenSpan
    probability: float
    predicted: bool


@dataclass(frozen=True)
class DocumentPrediction:
    doc_id: str
    scores: tuple[TokenScore, ...]
    predicted_starts: frozenset[int]
    invalid_boundary_offsets: tuple[int, ...]
    max_model_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tokenize_with_spans(text: str) -> list[TokenSpan]:
    tokens = [
        TokenSpan(index=index, text=match.group(), start=match.start(), end=match.end())
        for index, match in enumerate(TOKEN_RE.finditer(text))
    ]
    if not tokens:
        raise AuditError("Tokenization produced no tokens")
    cursor = 0
    for token in tokens:
        if text[cursor : token.start].strip():
            raise AuditError(f"Tokenizer skipped non-whitespace at character {cursor}")
        if text[token.start : token.end] != token.text:
            raise AuditError(f"Token span differs at character {token.start}")
        cursor = token.end
    if text[cursor:].strip():
        raise AuditError(f"Tokenizer skipped final text at character {cursor}")
    return tokens


def upstream_chunks(tokens: list[TokenSpan], max_words: int) -> list[list[TokenSpan]]:
    chunks: list[list[TokenSpan]] = []
    pending: list[TokenSpan] = []
    punctuation_seen = False
    for token in tokens:
        pending.append(token)
        if token.text in SENTENCE_END or len(pending) >= max_words:
            punctuation_seen = True
            while len(pending) > max_words:
                chunks.append(pending[:max_words])
                pending = pending[max_words:]
            if pending:
                chunks.append(pending)
                pending = []
    if pending:
        if punctuation_seen:
            chunks.append(pending)
        else:
            chunks.extend(
                pending[start : start + 8] for start in range(0, len(pending), 8)
            )
    if [token for chunk in chunks for token in chunk] != tokens:
        raise AuditError("SeCoRel chunking changed the canonical token sequence")
    return chunks


def predict_documents(
    audit,
    tokenizer,
    model,
    torch,
    device,
    *,
    boundary_label_id: int,
    max_length: int,
    max_words: int,
    batch_size: int,
) -> tuple[list[DocumentPrediction], dict[str, list[TokenSpan]]]:
    tokenized = {
        document.doc_id: tokenize_with_spans(document.raw_text)
        for document in audit.german_documents
    }
    items = [
        (document.doc_id, chunk)
        for document in audit.german_documents
        for chunk in upstream_chunks(tokenized[document.doc_id], max_words)
    ]
    grouped_scores: dict[str, list[TokenScore]] = {
        document.doc_id: [] for document in audit.german_documents
    }
    grouped_lengths: dict[str, list[int]] = {
        document.doc_id: [] for document in audit.german_documents
    }
    model.eval()
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        words = [[token.text for token in chunk] for _, chunk in batch]
        untruncated = tokenizer(
            words, is_split_into_words=True, add_special_tokens=True, truncation=False
        )
        lengths = [len(ids) for ids in untruncated["input_ids"]]
        if any(length > max_length for length in lengths):
            failed = [
                (doc_id, length)
                for (doc_id, _), length in zip(batch, lengths)
                if length > max_length
            ]
            raise AuditError(f"SeCoRel chunks exceed max_length={max_length}: {failed}")
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            padding=True,
        )
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            labels = logits.argmax(dim=-1)
        label_rows = labels.detach().cpu().tolist()
        probability_rows = probabilities[:, :, boundary_label_id].detach().cpu().tolist()
        for batch_index, ((doc_id, chunk), label_row, probability_row, length) in enumerate(
            zip(batch, label_rows, probability_rows, lengths)
        ):
            word_ids = encoded.word_ids(batch_index=batch_index)
            seen: set[int] = set()
            for model_index, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                grouped_scores[doc_id].append(
                    TokenScore(
                        token=chunk[word_id],
                        probability=float(probability_row[model_index]),
                        predicted=int(label_row[model_index]) == boundary_label_id,
                    )
                )
            if len(seen) != len(chunk):
                raise AuditError(f"SeCoRel wordpiece alignment lost tokens in {doc_id}")
            grouped_lengths[doc_id].append(length)
    predictions = []
    for document in audit.german_documents:
        scores = sorted(grouped_scores[document.doc_id], key=lambda score: score.token.index)
        if [score.token for score in scores] != tokenized[document.doc_id]:
            raise AuditError(f"SeCoRel score sequence differs for {document.doc_id}")
        valid_predicted = {
            score.token.start
            for score in scores
            if score.predicted
            and (
                score.token.start == 0
                or document.raw_text[score.token.start - 1].isspace()
            )
        }
        invalid = {
            score.token.start
            for score in scores
            if score.predicted
            and score.token.start != 0
            and not document.raw_text[score.token.start - 1].isspace()
        }
        predictions.append(
            DocumentPrediction(
                doc_id=document.doc_id,
                scores=tuple(scores),
                predicted_starts=frozenset({0} | valid_predicted),
                invalid_boundary_offsets=tuple(sorted(invalid)),
                max_model_tokens=max(grouped_lengths[document.doc_id]),
            )
        )
    return predictions, tokenized


def adu_for_offset(document, offset: int):
    return next(adu for adu in document.adus if adu.start <= offset < adu.end)


def boundary_rows(audit, predictions, config, *, constrained: bool) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        document = documents[prediction.doc_id]
        adu_starts = {adu.start for adu in document.adus}
        scores = {score.token.start: score for score in prediction.scores}
        starts = set(prediction.predicted_starts)
        if constrained:
            starts |= adu_starts
        for index, start in enumerate(sorted(starts), start=1):
            adu = adu_for_offset(document, start)
            predicted = start in prediction.predicted_starts and start != 0
            if start == 0:
                boundary_class = "document_start"
            elif start in adu_starts:
                boundary_class = "adu"
            else:
                boundary_class = "internal_edu"
            score = scores.get(start)
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "language": "de",
                    "char_offset": start,
                    "token_index": score.token.index if score else "",
                    "adu_id": adu.adu_id or "",
                    "edu_id": f"e{index}",
                    "boundary_class": boundary_class,
                    "source": config["model"]["name"] if predicted else "structural_constraint",
                    "status": "automatic" if predicted else "forced",
                    "confidence": f"{score.probability:.8f}" if predicted and score else "",
                    "model": config["model"]["name"] if predicted else "",
                    "run_id": config["run_id"],
                    "sameunit_affected": str(document.sameunit_affected).lower(),
                }
            )
    return rows


def score_rows(audit, predictions) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        adu_starts = {adu.start for adu in documents[prediction.doc_id].adus}
        for score in prediction.scores:
            rows.append(
                {
                    "doc_id": prediction.doc_id,
                    "token_index": score.token.index,
                    "char_offset": score.token.start,
                    "char_end": score.token.end,
                    "token_text": score.token.text,
                    "boundary_probability": f"{score.probability:.8f}",
                    "predicted_boundary": str(score.predicted).lower(),
                    "gold_adu_start": str(score.token.start in adu_starts).lower(),
                }
            )
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def strongest_scores(rows: list[dict[str, str]]) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["doc_id"], int(row["char_offset"]))
        scores[key] = max(scores.get(key, 0.0), float(row["boundary_probability"]))
    return scores


def comparison_rows(audit, predictions) -> list[dict[str, object]]:
    eduseg_boundaries = read_tsv(
        REPO_ROOT / "derived" / "boundaries" / "de_eduseg_de_document_v1_raw.tsv"
    )
    eduseg_starts = {
        (row["doc_id"], int(row["char_offset"]))
        for row in eduseg_boundaries
        if row["boundary_class"] == "internal_edu"
    }
    eduseg_scores = strongest_scores(
        read_tsv(REPO_ROOT / "derived" / "scores" / "de_eduseg_de_document_v1.tsv")
    )
    secorel_scores = {
        (prediction.doc_id, score.token.start): score.probability
        for prediction in predictions
        for score in prediction.scores
    }
    documents = {document.doc_id: document for document in audit.german_documents}
    secorel_starts = {
        (prediction.doc_id, start)
        for prediction in predictions
        for start in prediction.predicted_starts
        if start != 0
        and start not in {adu.start for adu in documents[prediction.doc_id].adus}
    }
    rows: list[dict[str, object]] = []
    for doc_id, start in sorted(eduseg_starts | secorel_starts):
        document = documents[doc_id]
        adu = adu_for_offset(document, start)
        rows.append(
            {
                "doc_id": doc_id,
                "adu_id": adu.adu_id or "",
                "char_offset": start,
                "eduseg_de_predicted": str((doc_id, start) in eduseg_starts).lower(),
                "secorel_predicted": str((doc_id, start) in secorel_starts).lower(),
                "eduseg_de_probability": f"{eduseg_scores[(doc_id, start)]:.8f}",
                "secorel_probability": f"{secorel_scores[(doc_id, start)]:.8f}",
                "left_context": document.raw_text[max(0, start - 100) : start].strip(),
                "right_context": document.raw_text[start : start + 100].strip(),
            }
        )
    return rows


def make_summary(audit, predictions, comparisons) -> dict[str, object]:
    known = 0
    recovered = 0
    internal = 0
    for document, prediction in zip(audit.german_documents, predictions):
        gold = {adu.start for adu in document.adus if adu.start != 0}
        predicted = set(prediction.predicted_starts) - {0}
        known += len(gold)
        recovered += len(gold & predicted)
        internal += len(predicted - gold)
    shared = sum(
        row["eduseg_de_predicted"] == "true" and row["secorel_predicted"] == "true"
        for row in comparisons
    )
    eduseg_only = sum(
        row["eduseg_de_predicted"] == "true" and row["secorel_predicted"] == "false"
        for row in comparisons
    )
    secorel_only = sum(
        row["eduseg_de_predicted"] == "false" and row["secorel_predicted"] == "true"
        for row in comparisons
    )
    return {
        "documents": len(predictions),
        "gold_noninitial_adu_starts": known,
        "recovered_noninitial_adu_starts": recovered,
        "adu_boundary_recall": recovered / known,
        "secorel_internal_proposals": internal,
        "eduseg_de_internal_proposals": shared + eduseg_only,
        "shared_internal_proposals": shared,
        "eduseg_de_only": eduseg_only,
        "secorel_only": secorel_only,
        "internal_boundary_jaccard": shared / (shared + eduseg_only + secorel_only),
        "internal_boundary_f1_agreement": (
            2 * shared / (2 * shared + eduseg_only + secorel_only)
        ),
        "max_model_tokens_per_chunk": max(
            prediction.max_model_tokens for prediction in predictions
        ),
        "invalid_token_boundary_predictions": sum(
            len(prediction.invalid_boundary_offsets) for prediction in predictions
        ),
        "invalid_token_boundaries": [
            {
                "doc_id": prediction.doc_id,
                "char_offsets": list(prediction.invalid_boundary_offsets),
            }
            for prediction in predictions
            if prediction.invalid_boundary_offsets
        ],
        "interpretation": "System agreement and ADU recall, not German EDU accuracy.",
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        config = load_config(args.config)
        model_hashes = verify_model(args.model_dir, config)
        audit = build_audit()
        try:
            import torch
            import transformers
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise AuditError("Install torch and transformers to run SeCoRel") from exc
        device = select_device(args.device, torch)
        batch_size = args.batch_size or int(config["inference"]["batch_size"])
        if batch_size <= 0:
            raise AuditError(f"Batch size must be positive, found {batch_size}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if not tokenizer.is_fast:
            raise AuditError("SeCoRel requires a fast tokenizer for wordpiece alignment")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir, local_files_only=True
        ).to(device)
        boundary_label_id = int(config["model"]["boundary_label_id"])
        outside_label_id = int(config["model"]["outside_label_id"])
        if {outside_label_id, boundary_label_id} != set(range(model.config.num_labels)):
            raise AuditError("SeCoRel model/config label mapping differs")
        predictions, tokenized = predict_documents(
            audit,
            tokenizer,
            model,
            torch,
            device,
            boundary_label_id=boundary_label_id,
            max_length=int(config["model"]["max_length"]),
            max_words=int(config["model"]["max_words"]),
            batch_size=batch_size,
        )
        if bool(config["inference"]["require_all_reference_boundaries_at_token_starts"]):
            for document in audit.german_documents:
                token_starts = {token.start for token in tokenized[document.doc_id]}
                reference = {adu.start for adu in document.adus}
                if not reference <= token_starts:
                    raise AuditError(f"ADU starts are not token starts in {document.doc_id}")
        raw_rows = boundary_rows(audit, predictions, config, constrained=False)
        constrained_rows = boundary_rows(audit, predictions, config, constrained=True)
        scores = score_rows(audit, predictions)
        comparisons = comparison_rows(audit, predictions)
        summary = make_summary(audit, predictions, comparisons)
        prepare_output(args.output_dir, args.overwrite)
        raw_dir = args.output_dir / "raw_edus"
        constrained_dir = args.output_dir / "constrained_edus"
        raw_dir.mkdir()
        constrained_dir.mkdir()
        prediction_by_doc = {prediction.doc_id: prediction for prediction in predictions}
        for document in audit.german_documents:
            prediction = prediction_by_doc[document.doc_id]
            adu_starts = {adu.start for adu in document.adus}
            raw = segment_text(document.raw_text, prediction.predicted_starts)
            constrained = segment_text(
                document.raw_text, set(prediction.predicted_starts) | adu_starts
            )
            (raw_dir / f"{document.doc_id}.edus").write_text(
                "\n".join(raw) + "\n", encoding="utf-8"
            )
            (constrained_dir / f"{document.doc_id}.edus").write_text(
                "\n".join(constrained) + "\n", encoding="utf-8"
            )
        write_tsv(args.output_dir / "raw_boundaries.tsv", raw_rows, BOUNDARY_FIELDS)
        write_tsv(
            args.output_dir / "constrained_boundaries.tsv",
            constrained_rows,
            BOUNDARY_FIELDS,
        )
        write_tsv(args.output_dir / "boundary_scores.tsv", scores, SCORE_FIELDS)
        write_tsv(args.output_dir / "model_comparison.tsv", comparisons, COMPARISON_FIELDS)
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
            "tokenization": {
                "name": config["tokenization"],
                "regex": TOKEN_RE.pattern,
                "sentence_end_tokens": sorted(SENTENCE_END),
            },
            "inference": {
                **config["inference"],
                "chunking": config["model_chunking"],
                "device": str(device),
                "batch_size": batch_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "comparison": {
                "run_id": config["comparison_run"],
                "boundary_sha256": sha256_file(
                    REPO_ROOT
                    / "derived"
                    / "boundaries"
                    / "de_eduseg_de_document_v1_raw.tsv"
                ),
            },
            "outputs": {
                "raw_edu_files": len(list(raw_dir.glob("*.edus"))),
                "constrained_edu_files": len(list(constrained_dir.glob("*.edus"))),
                "raw_boundary_rows": len(raw_rows),
                "constrained_boundary_rows": len(constrained_rows),
                "score_rows": len(scores),
                "comparison_rows": len(comparisons),
                "files": output_hashes(args.output_dir),
            },
            "summary": summary,
        }
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (AuditError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"SECOREL RUN FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
