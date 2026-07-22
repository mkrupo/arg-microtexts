#!/usr/bin/env python3
"""Run the released German EduSeg model with offset-preserving provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import BOUNDARY_FIELDS, REPO_ROOT, AuditError, build_audit, write_json


DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "eduseg_de_document_v1.toml"
DEFAULT_OUTPUT = REPO_ROOT / "work" / "runs" / "eduseg_de_document_v1"
SCORE_FIELDS = [
    "doc_id",
    "model_token_index",
    "char_offset",
    "char_end",
    "token_text",
    "boundary_probability",
    "predicted_boundary",
    "gold_adu_start",
]


@dataclass(frozen=True)
class CandidateScore:
    model_token_index: int
    start: int
    end: int
    token_text: str
    probability: float
    predicted: bool


@dataclass(frozen=True)
class DocumentPrediction:
    doc_id: str
    text: str
    scores: tuple[CandidateScore, ...]
    predicted_starts: frozenset[int]
    token_count: int
    invalid_boundary_offsets: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, help="Override the committed batch size.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise AuditError("Python 3.10 requires the optional 'tomli' package") from exc
    with path.open("rb") as stream:
        return tomllib.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_dir: Path, config: dict[str, object]) -> dict[str, str]:
    configured = config["model"]["sha256"]
    files = {
        "config_json": model_dir / "config.json",
        "model_safetensors": model_dir / "model.safetensors",
        "tokenizer_json": model_dir / "tokenizer.json",
        "sentencepiece_bpe_model": model_dir / "sentencepiece.bpe.model",
        "special_tokens_map_json": model_dir / "special_tokens_map.json",
        "tokenizer_config_json": model_dir / "tokenizer_config.json",
    }
    actual: dict[str, str] = {}
    for key, path in files.items():
        if not path.is_file():
            raise AuditError(f"Required model file is missing: {path}")
        actual[key] = sha256_file(path)
        expected = configured[key]
        if actual[key] != expected:
            raise AuditError(
                f"Model hash mismatch for {path.name}: expected {expected}, found {actual[key]}"
            )
    return actual


def select_device(requested: str, torch):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise AuditError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_path(path: Path) -> str:
    """Return a public, repository-relative path without leaking local directories."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise AuditError(f"Configuration must be inside the repository: {path}") from exc


def output_hashes(root: Path) -> list[dict[str, str]]:
    """Hash every completed run artifact except the manifest that embeds this list."""
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.name != "manifest.json"
    ]


def is_text_boundary(text: str, start: int) -> bool:
    return start == 0 or (0 < start <= len(text) and text[start - 1].isspace())


def segment_text(text: str, starts: Iterable[int]) -> list[str]:
    ordered = sorted(set(starts) | {0})
    if any(start < 0 or start >= len(text) for start in ordered):
        raise AuditError(f"Boundary outside document of length {len(text)}: {ordered}")
    if any(not is_text_boundary(text, start) for start in ordered):
        invalid = [start for start in ordered if not is_text_boundary(text, start)]
        raise AuditError(f"Boundaries are not text starts: {invalid}")
    edus = [text[start:end].strip() for start, end in zip(ordered, ordered[1:] + [len(text)])]
    if any(not edu for edu in edus):
        raise AuditError("Segmentation produced an empty EDU")
    if " ".join(edus) != text:
        raise AuditError("Segmented EDUs do not reconstruct the raw text")
    return edus


def predict_documents(
    *,
    documents,
    tokenizer,
    model,
    torch,
    device,
    boundary_label_id: int,
    max_length: int,
    batch_size: int,
    require_text_boundary: bool,
) -> list[DocumentPrediction]:
    predictions: list[DocumentPrediction] = []
    model.eval()
    for batch_start in range(0, len(documents), batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        texts = [document.raw_text for document in batch]
        untruncated = tokenizer(texts, add_special_tokens=True, truncation=False)
        lengths = [len(ids) for ids in untruncated["input_ids"]]
        if any(length > max_length for length in lengths):
            too_long = [
                (document.doc_id, length)
                for document, length in zip(batch, lengths)
                if length > max_length
            ]
            raise AuditError(f"Documents exceed max_length={max_length}: {too_long}")

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            label_ids = logits.argmax(dim=-1)

        input_ids = encoded["input_ids"].detach().cpu().tolist()
        offsets_list = offsets.detach().cpu().tolist()
        labels_list = label_ids.detach().cpu().tolist()
        probabilities_list = probabilities[:, :, boundary_label_id].detach().cpu().tolist()

        batch_predictions = zip(
            batch,
            input_ids,
            offsets_list,
            labels_list,
            probabilities_list,
            lengths,
        )
        for (
            document,
            token_ids,
            token_offsets,
            token_labels,
            token_probabilities,
            token_count,
        ) in batch_predictions:
            scores: list[CandidateScore] = []
            predicted_starts = {0}
            invalid: list[int] = []
            token_predictions = zip(
                token_ids, token_offsets, token_labels, token_probabilities
            )
            for model_token_index, (token_id, (start, end), label_id, probability) in enumerate(
                token_predictions
            ):
                start = int(start)
                end = int(end)
                if start == end:
                    continue
                predicted = int(label_id) == boundary_label_id
                valid_start = is_text_boundary(document.raw_text, start)
                if predicted and require_text_boundary and not valid_start:
                    invalid.append(start)
                    continue
                if not valid_start:
                    continue
                scores.append(
                    CandidateScore(
                        model_token_index=model_token_index,
                        start=start,
                        end=end,
                        token_text=tokenizer.convert_ids_to_tokens(int(token_id)),
                        probability=float(probability),
                        predicted=predicted,
                    )
                )
                if predicted:
                    predicted_starts.add(start)
            predictions.append(
                DocumentPrediction(
                    doc_id=document.doc_id,
                    text=document.raw_text,
                    scores=tuple(scores),
                    predicted_starts=frozenset(predicted_starts),
                    token_count=token_count,
                    invalid_boundary_offsets=tuple(sorted(set(invalid))),
                )
            )
    return predictions


def adu_for_offset(document, offset: int):
    for adu in document.adus:
        if adu.start <= offset < adu.end:
            return adu
    if offset == len(document.raw_text):
        return document.adus[-1]
    raise AuditError(f"No ADU contains {document.doc_id} character offset {offset}")


def boundary_rows(
    *,
    audit,
    predictions: list[DocumentPrediction],
    config: dict[str, object],
    constrained: bool,
) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        document = documents[prediction.doc_id]
        adu_starts = {adu.start for adu in document.adus}
        # SentencePiece can emit two model tokens with the same raw-text start
        # (for example, a standalone whitespace marker plus a lexical piece).
        # A textual boundary exists if either token predicts B, so retain the
        # strongest B probability at each character start for boundary metadata.
        score_by_start: dict[int, CandidateScore] = {}
        for score in prediction.scores:
            previous = score_by_start.get(score.start)
            if previous is None or score.probability > previous.probability:
                score_by_start[score.start] = score
        starts = set(prediction.predicted_starts)
        if constrained:
            starts |= adu_starts
        for index, start in enumerate(sorted(starts), start=1):
            adu = adu_for_offset(document, start)
            if start == 0:
                boundary_class = "document_start"
            elif start in adu_starts:
                boundary_class = "adu"
            else:
                boundary_class = "internal_edu"
            predicted = start in prediction.predicted_starts and start != 0
            score = score_by_start.get(start)
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "language": "de",
                    "char_offset": start,
                    "token_index": "",
                    "adu_id": adu.adu_id or "",
                    "edu_id": f"e{index}",
                    "boundary_class": boundary_class,
                    "source": (
                        config["model"]["name"] if predicted else "structural_constraint"
                    ),
                    "status": "automatic" if predicted else "forced",
                    "confidence": f"{score.probability:.8f}" if score else "",
                    "model": (
                        f"{config['model']['name']}@{config['model']['revision']}"
                        if predicted
                        else ""
                    ),
                    "run_id": config["run_id"],
                    "sameunit_affected": str(document.sameunit_affected).lower(),
                }
            )
    return rows


def score_rows(audit, predictions: list[DocumentPrediction]) -> list[dict[str, object]]:
    documents = {document.doc_id: document for document in audit.german_documents}
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        adu_starts = {adu.start for adu in documents[prediction.doc_id].adus}
        for score in prediction.scores:
            rows.append(
                {
                    "doc_id": prediction.doc_id,
                    "model_token_index": score.model_token_index,
                    "char_offset": score.start,
                    "char_end": score.end,
                    "token_text": score.token_text,
                    "boundary_probability": f"{score.probability:.8f}",
                    "predicted_boundary": str(score.predicted).lower(),
                    "gold_adu_start": str(score.start in adu_starts).lower(),
                }
            )
    return rows


def make_summary(audit, predictions: list[DocumentPrediction]) -> dict[str, object]:
    documents = {document.doc_id: document for document in audit.german_documents}
    known = 0
    recovered = 0
    internal = 0
    after_terminal = 0
    raw_noninitial = 0
    per_group: dict[str, dict[str, int]] = {}
    missing: list[dict[str, object]] = []
    for prediction in predictions:
        document = documents[prediction.doc_id]
        gold = {adu.start for adu in document.adus if adu.start != 0}
        predicted = set(prediction.predicted_starts) - {0}
        hits = gold & predicted
        additions = predicted - gold
        after_terminal += sum(
            document.raw_text[:start].rstrip().endswith((".", "!", "?", "…"))
            for start in additions
        )
        group = prediction.doc_id.removeprefix("micro_")[0]
        group_counts = per_group.setdefault(
            group,
            {
                "documents": 0,
                "gold_adu_starts": 0,
                "recovered_adu_starts": 0,
                "internal_proposals": 0,
            },
        )
        group_counts["documents"] += 1
        group_counts["gold_adu_starts"] += len(gold)
        group_counts["recovered_adu_starts"] += len(hits)
        group_counts["internal_proposals"] += len(additions)
        known += len(gold)
        recovered += len(hits)
        internal += len(additions)
        raw_noninitial += len(predicted)
        for start in sorted(gold - predicted):
            adu = adu_for_offset(document, start)
            missing.append(
                {
                    "doc_id": document.doc_id,
                    "char_offset": start,
                    "adu_id": adu.adu_id,
                    "text": adu.text,
                }
            )
    for values in per_group.values():
        values["adu_recall_ppm"] = round(
            1_000_000 * values["recovered_adu_starts"] / values["gold_adu_starts"]
        )
    return {
        "documents": len(predictions),
        "gold_noninitial_adu_starts": known,
        "recovered_noninitial_adu_starts": recovered,
        "adu_boundary_recall": recovered / known,
        "raw_noninitial_predicted_starts": raw_noninitial,
        "internal_boundary_proposals": internal,
        "internal_proposals_after_terminal_punctuation": after_terminal,
        "internal_proposals_sentence_internal": internal - after_terminal,
        "missing_adu_boundaries": missing,
        "per_group": per_group,
        "max_model_tokens": max(prediction.token_count for prediction in predictions),
        "invalid_subword_boundary_predictions": sum(
            len(prediction.invalid_boundary_offsets) for prediction in predictions
        ),
        "invalid_subword_boundaries": [
            {
                "doc_id": prediction.doc_id,
                "char_offsets": list(prediction.invalid_boundary_offsets),
            }
            for prediction in predictions
            if prediction.invalid_boundary_offsets
        ],
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise AuditError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


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
            raise AuditError("Install torch and transformers to run EduSeg inference") from exc

        device = select_device(args.device, torch)
        batch_size = args.batch_size or int(config["inference"]["batch_size"])
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if not tokenizer.is_fast:
            raise AuditError("EduSeg inference requires a fast tokenizer for character offsets")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir, local_files_only=True
        ).to(device)
        boundary_label_id = int(config["model"]["boundary_label_id"])
        inside_label_id = int(config["model"]["inside_label_id"])
        if {inside_label_id, boundary_label_id} != set(range(model.config.num_labels)):
            raise AuditError(
                f"Model/config label mismatch: model has {model.config.num_labels} labels, "
                f"inside={inside_label_id}, boundary={boundary_label_id}"
            )
        if batch_size <= 0:
            raise AuditError(f"Batch size must be positive, found {batch_size}")

        predictions = predict_documents(
            documents=list(audit.german_documents),
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            boundary_label_id=boundary_label_id,
            max_length=int(config["model"]["max_length"]),
            batch_size=batch_size,
            require_text_boundary=bool(config["inference"]["require_text_boundary"]),
        )
        prepare_output(args.output_dir, args.overwrite)
        raw_dir = args.output_dir / "raw_edus"
        constrained_dir = args.output_dir / "constrained_edus"
        raw_dir.mkdir()
        constrained_dir.mkdir()
        documents = {document.doc_id: document for document in audit.german_documents}
        for prediction in predictions:
            adu_starts = {adu.start for adu in documents[prediction.doc_id].adus}
            raw = segment_text(prediction.text, prediction.predicted_starts)
            constrained = segment_text(
                prediction.text, set(prediction.predicted_starts) | adu_starts
            )
            (raw_dir / f"{prediction.doc_id}.edus").write_text(
                "\n".join(raw) + "\n", encoding="utf-8"
            )
            (constrained_dir / f"{prediction.doc_id}.edus").write_text(
                "\n".join(constrained) + "\n", encoding="utf-8"
            )

        raw_rows = boundary_rows(
            audit=audit, predictions=predictions, config=config, constrained=False
        )
        constrained_rows = boundary_rows(
            audit=audit, predictions=predictions, config=config, constrained=True
        )
        all_scores = score_rows(audit, predictions)
        write_tsv(args.output_dir / "raw_boundaries.tsv", raw_rows, BOUNDARY_FIELDS)
        write_tsv(
            args.output_dir / "constrained_boundaries.tsv", constrained_rows, BOUNDARY_FIELDS
        )
        write_tsv(args.output_dir / "boundary_scores.tsv", all_scores, SCORE_FIELDS)
        summary = make_summary(audit, predictions)
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
            "input_corpus_tree": audit.original_corpus_tree,
            "model": {
                **config["model"],
                "verified_sha256": model_hashes,
            },
            "inference": {
                **config["inference"],
                "device": str(device),
                "batch_size": batch_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "outputs": {
                "raw_edu_files": len(list(raw_dir.glob("*.edus"))),
                "constrained_edu_files": len(list(constrained_dir.glob("*.edus"))),
                "raw_boundary_rows": len(raw_rows),
                "constrained_boundary_rows": len(constrained_rows),
                "score_rows": len(all_scores),
                "files": output_hashes(args.output_dir),
            },
            "summary": summary,
        }
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (AuditError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"EDUSEG RUN FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
