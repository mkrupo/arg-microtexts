#!/usr/bin/env python3
"""Build the deterministic browser dataset for the EDU explorer website."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import REPO_ROOT, UnitSpan, build_audit


DEFAULT_OUTPUT = REPO_ROOT / "public" / "data" / "edu-explorer.json"
BOUNDARY_PATHS = {
    "eduseg_document": REPO_ROOT
    / "derived"
    / "boundaries"
    / "de_eduseg_de_document_v1_raw.tsv",
    "eduseg_adu": REPO_ROOT
    / "derived"
    / "boundaries"
    / "de_eduseg_de_adu_context_v1.tsv",
    "secorel": REPO_ROOT
    / "derived"
    / "boundaries"
    / "de_secorel_disrpt_sentence_chunks_v1_raw.tsv",
}
SCORE_PATHS = {
    "eduseg_document": REPO_ROOT
    / "derived"
    / "scores"
    / "de_eduseg_de_document_v1.tsv",
    "eduseg_adu": REPO_ROOT
    / "derived"
    / "scores"
    / "de_eduseg_de_adu_context_v1.tsv",
    "secorel": REPO_ROOT
    / "derived"
    / "scores"
    / "de_secorel_disrpt_sentence_chunks_v1.tsv",
}
SUMMARY_PATHS = {
    "eduseg_document": REPO_ROOT
    / "experiments"
    / "results"
    / "eduseg_de_document_v1"
    / "summary.json",
    "eduseg_adu": REPO_ROOT
    / "experiments"
    / "results"
    / "eduseg_de_adu_context_v1"
    / "summary.json",
    "secorel": REPO_ROOT
    / "experiments"
    / "results"
    / "secorel_disrpt_sentence_chunks_v1"
    / "summary.json",
}
MANIFEST_PATHS = {
    key: path.with_name("manifest.json") for key, path in SUMMARY_PATHS.items()
}

MODEL_LABELS = {
    "eduseg_document": "EduSeg · document",
    "eduseg_adu": "EduSeg · per ADU",
    "secorel": "SeCoRel · sentence chunks",
}

CURATED_CASES = [
    {
        "docId": "micro_b001",
        "aduId": "a5",
        "offset": 405,
        "kind": "likely-oversegmentation",
        "eyebrow": "Agreement is not correctness",
        "title": "A shared coordination split",
        "note": (
            "All three runs split before “und Vorreiter …”. The shared subject and "
            "modal construction motivate the current expert working analysis of one "
            "EDU. This is a review hypothesis, not adjudicated German gold."
        ),
    },
    {
        "docId": "micro_b014",
        "aduId": "a4",
        "offset": 315,
        "kind": "plausible-refinement",
        "eyebrow": "A plausible refinement",
        "title": "Two questions inside one ADU",
        "note": (
            "The only EduSeg document proposal after terminal punctuation separates "
            "two questions. Both EduSeg contexts and SeCoRel strongly support it."
        ),
    },
    {
        "docId": "micro_d09",
        "aduId": "a3",
        "offset": 75,
        "kind": "shared-miss",
        "eyebrow": "Known-boundary diagnostic",
        "title": "Both document models miss an ADU start",
        "note": (
            "EduSeg document context and SeCoRel both score this locked ADU start "
            "below threshold. The constrained exports restore it structurally."
        ),
    },
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_value(value: str) -> bool:
    return value.lower() == "true"


def rounded_probability(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    return round(float(value), 6)


def group_name(doc_id: str) -> str:
    return doc_id.removeprefix("micro_")[0]


def containing_span(spans: Iterable[UnitSpan], offset: int) -> UnitSpan:
    span_list = list(spans)
    for span in span_list:
        if span.start <= offset < span.end:
            return span
    if offset == span_list[-1].end:
        return span_list[-1]
    raise ValueError(f"No unit contains character offset {offset}")


def visible_context(text: str, offset: int, radius: int = 72) -> dict[str, str]:
    return {
        "left": text[max(0, offset - radius) : offset].strip(),
        "right": text[offset : min(len(text), offset + radius)].strip(),
    }


def load_predictions() -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Return model -> document -> offset -> published boundary metadata."""
    result: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model_key, path in BOUNDARY_PATHS.items():
        documents: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in read_tsv(path):
            offset = int(row["char_offset"])
            is_model_prediction = (
                row["status"] == "automatic"
                and row["source"] not in {"structural_constraint", ""}
            )
            documents[row["doc_id"]][offset] = {
                "predicted": is_model_prediction,
                "boundaryClass": row["boundary_class"],
                "confidence": rounded_probability(row["confidence"]),
                "status": row["status"],
            }
        result[model_key] = dict(documents)
    return result


def load_scores() -> dict[str, dict[str, dict[int, float]]]:
    """Take the maximum score where several model tokens share a text start."""
    result: dict[str, dict[str, dict[int, float]]] = {}
    for model_key, path in SCORE_PATHS.items():
        documents: dict[str, dict[int, float]] = defaultdict(dict)
        for row in read_tsv(path):
            offset = int(row["char_offset"])
            probability = float(row["boundary_probability"])
            previous = documents[row["doc_id"]].get(offset)
            if previous is None or probability > previous:
                documents[row["doc_id"]][offset] = probability
        result[model_key] = {
            doc_id: {offset: round(value, 6) for offset, value in scores.items()}
            for doc_id, scores in documents.items()
        }
    return result


def layer_at(
    model_key: str,
    doc_id: str,
    offset: int,
    gold_start: bool,
    predictions: dict[str, dict[str, dict[int, dict[str, Any]]]],
    scores: dict[str, dict[str, dict[int, float]]],
) -> dict[str, Any]:
    record = predictions[model_key].get(doc_id, {}).get(offset)
    predicted = bool(record and record["predicted"])
    score = scores[model_key].get(doc_id, {}).get(offset)

    # Per-ADU sequence starts are structural and their scores are not comparable
    # with genuinely evaluated within-ADU positions.
    if model_key == "eduseg_adu" and gold_start:
        score = None

    return {
        "predicted": predicted,
        "score": score,
        "evidence": (
            "predicted"
            if predicted
            else "below-threshold"
            if score is not None
            else "not-comparable"
        ),
    }


def boundary_record(
    doc_id: str,
    text: str,
    adus: tuple[UnitSpan, ...],
    offset: int,
    gold_offsets: set[int],
    predictions: dict[str, dict[str, dict[int, dict[str, Any]]]],
    scores: dict[str, dict[str, dict[int, float]]],
) -> dict[str, Any]:
    span = containing_span(adus, offset)
    gold_start = offset in gold_offsets
    return {
        "offset": offset,
        "aduId": span.adu_id,
        "gold": gold_start,
        "boundaryClass": "document-start"
        if offset == 0
        else "adu"
        if gold_start
        else "internal",
        "afterTerminalPunctuation": (
            bool(text[:offset].rstrip())
            and text[:offset].rstrip().endswith((".", "?", "!"))
        ),
        "context": visible_context(text, offset),
        "layers": {
            model_key: layer_at(
                model_key,
                doc_id,
                offset,
                gold_start,
                predictions,
                scores,
            )
            for model_key in MODEL_LABELS
        },
    }


def english_units_for_adu(
    adu: UnitSpan, english_edus: tuple[UnitSpan, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "id": edu.unit_id,
            "start": edu.start,
            "end": edu.end,
            "text": edu.text,
            "internal": edu.start != adu.start,
        }
        for edu in english_edus
        if edu.adu_id == adu.adu_id
    ]


def build_dataset() -> dict[str, Any]:
    audit = build_audit()
    predictions = load_predictions()
    scores = load_scores()
    summaries = {key: read_json(path) for key, path in SUMMARY_PATHS.items()}
    manifests = {key: read_json(path) for key, path in MANIFEST_PATHS.items()}
    english_by_id = {document.doc_id: document for document in audit.english_documents}
    records_by_id = {document.doc_id: document for document in audit.documents}

    documents = []
    aggregate_group_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "documents": 0,
            "edusegDocument": 0,
            "edusegAdu": 0,
            "secorel": 0,
        }
    )

    for german in audit.german_documents:
        english = english_by_id[german.doc_id]
        record = records_by_id[german.doc_id]
        gold_offsets = {adu.start for adu in german.adus}
        candidate_offsets = set(gold_offsets)
        for model_key in MODEL_LABELS:
            candidate_offsets.update(
                offset
                for offset, value in predictions[model_key]
                .get(german.doc_id, {})
                .items()
                if value["predicted"]
            )

        boundaries = [
            boundary_record(
                german.doc_id,
                german.raw_text,
                german.adus,
                offset,
                gold_offsets,
                predictions,
                scores,
            )
            for offset in sorted(candidate_offsets)
        ]
        internal = [boundary for boundary in boundaries if not boundary["gold"]]
        model_sets = {
            model_key: {
                boundary["offset"]
                for boundary in internal
                if boundary["layers"][model_key]["predicted"]
            }
            for model_key in MODEL_LABELS
        }
        doc_vs_secorel_union = model_sets["eduseg_document"] | model_sets["secorel"]
        doc_vs_secorel_shared = model_sets["eduseg_document"] & model_sets["secorel"]
        all_three_shared = set.intersection(*model_sets.values())
        any_disagreement = {
            boundary["offset"]
            for boundary in internal
            if len(
                {
                    boundary["layers"][model_key]["predicted"]
                    for model_key in MODEL_LABELS
                }
            )
            > 1
        }

        missed_gold: dict[str, list[int]] = {}
        for model_key in ("eduseg_document", "secorel"):
            missed_gold[model_key] = [
                offset
                for offset in sorted(gold_offsets - {0})
                if not predictions[model_key]
                .get(german.doc_id, {})
                .get(offset, {})
                .get("predicted", False)
            ]

        group = group_name(german.doc_id)
        aggregate_group_counts[group]["documents"] += 1
        aggregate_group_counts[group]["edusegDocument"] += len(
            model_sets["eduseg_document"]
        )
        aggregate_group_counts[group]["edusegAdu"] += len(model_sets["eduseg_adu"])
        aggregate_group_counts[group]["secorel"] += len(model_sets["secorel"])

        documents.append(
            {
                "id": german.doc_id,
                "group": group,
                "sameUnitAffected": german.sameunit_affected,
                "german": {
                    "text": german.raw_text,
                    "adus": [
                        {
                            "id": adu.adu_id,
                            "sourceUnitId": adu.unit_id,
                            "start": adu.start,
                            "end": adu.end,
                            "text": adu.text,
                        }
                        for adu in german.adus
                    ],
                },
                "english": {
                    "text": english.raw_text,
                    "adus": [
                        {
                            "id": adu.adu_id,
                            "sourceUnitId": adu.unit_id,
                            "start": adu.start,
                            "end": adu.end,
                            "text": adu.text,
                            "edus": english_units_for_adu(adu, english.edus),
                        }
                        for adu in english.original_adus
                    ],
                },
                "boundaries": boundaries,
                "stats": {
                    "aduCount": len(german.adus),
                    "englishEduCount": record.english_edu_count,
                    "englishInternalGold": record.internal_boundary_count,
                    "edusegDocument": len(model_sets["eduseg_document"]),
                    "edusegAdu": len(model_sets["eduseg_adu"]),
                    "secorel": len(model_sets["secorel"]),
                    "edusegSecorelShared": len(doc_vs_secorel_shared),
                    "edusegSecorelUnion": len(doc_vs_secorel_union),
                    "allThreeShared": len(all_three_shared),
                    "disagreements": len(any_disagreement),
                    "missedAduEdUseg": missed_gold["eduseg_document"],
                    "missedAduSecorel": missed_gold["secorel"],
                    "reviewScore": (
                        len(any_disagreement) * 3
                        + len(missed_gold["eduseg_document"]) * 2
                        + len(missed_gold["secorel"]) * 2
                        + len(all_three_shared)
                    ),
                },
            }
        )

    curated = []
    documents_by_id = {document["id"]: document for document in documents}
    for case in CURATED_CASES:
        document = documents_by_id[case["docId"]]
        boundary = next(
            item for item in document["boundaries"] if item["offset"] == case["offset"]
        )
        curated.append(
            {
                **case,
                "context": boundary["context"],
                "layers": boundary["layers"],
            }
        )

    dataset = {
        "schemaVersion": 1,
        "generatedFrom": {
            "originalCorpusTree": audit.original_corpus_tree,
            "multilayerCommit": audit.source_commit,
            "runProjectCommits": {
                key: manifest["source_run"]["project_commit"]
                for key, manifest in manifests.items()
            },
            "sources": [
                str(path.relative_to(REPO_ROOT))
                for path in [
                    *BOUNDARY_PATHS.values(),
                    *SCORE_PATHS.values(),
                    *SUMMARY_PATHS.values(),
                    *MANIFEST_PATHS.values(),
                ]
            ],
        },
        "summary": {
            "corpus": audit.totals,
            "models": {
                "edusegDocument": {
                    "label": MODEL_LABELS["eduseg_document"],
                    "protocol": "One complete German microtext per model input.",
                    "internalProposals": summaries["eduseg_document"][
                        "internal_boundary_proposals"
                    ],
                    "recoveredAduStarts": summaries["eduseg_document"][
                        "recovered_noninitial_adu_starts"
                    ],
                    "goldAduStarts": summaries["eduseg_document"][
                        "gold_noninitial_adu_starts"
                    ],
                },
                "edusegAdu": {
                    "label": MODEL_LABELS["eduseg_adu"],
                    "protocol": (
                        "Each locked ADU is an independent model input; only internal "
                        "predictions are compared."
                    ),
                    "internalProposals": summaries["eduseg_adu"][
                        "adu_context_internal_proposals"
                    ],
                },
                "secorel": {
                    "label": MODEL_LABELS["secorel"],
                    "protocol": (
                        "DISRPT-like tokens in sentence-terminal/280-word chunks, "
                        "matching the model interface."
                    ),
                    "internalProposals": summaries["secorel"][
                        "secorel_internal_proposals"
                    ],
                    "recoveredAduStarts": summaries["secorel"][
                        "recovered_noninitial_adu_starts"
                    ],
                    "goldAduStarts": summaries["secorel"]["gold_noninitial_adu_starts"],
                },
            },
            "agreements": {
                "edusegContexts": {
                    "shared": summaries["eduseg_adu"]["shared_internal_proposals"],
                    "documentOnly": summaries["eduseg_adu"]["document_context_only"],
                    "aduOnly": summaries["eduseg_adu"]["adu_context_only"],
                    "f1": round(
                        summaries["eduseg_adu"]["internal_boundary_f1_agreement"], 4
                    ),
                    "jaccard": round(
                        summaries["eduseg_adu"]["internal_boundary_jaccard"], 4
                    ),
                },
                "edusegSecorel": {
                    "shared": summaries["secorel"]["shared_internal_proposals"],
                    "edusegOnly": summaries["secorel"]["eduseg_de_only"],
                    "secorelOnly": summaries["secorel"]["secorel_only"],
                    "f1": round(
                        summaries["secorel"]["internal_boundary_f1_agreement"], 4
                    ),
                    "jaccard": round(
                        summaries["secorel"]["internal_boundary_jaccard"], 4
                    ),
                },
            },
            "groups": dict(sorted(aggregate_group_counts.items())),
            "curatedCases": curated,
        },
        "documents": documents,
    }

    assert len(documents) == 112
    assert sum(document["stats"]["edusegDocument"] for document in documents) == 118
    assert sum(document["stats"]["edusegAdu"] for document in documents) == 121
    assert sum(document["stats"]["secorel"] for document in documents) == 133
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing output differs from a fresh deterministic build.",
    )
    args = parser.parse_args()

    rendered = json.dumps(
        build_dataset(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) + "\n"
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing generated dataset: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Generated dataset is stale: {args.output}")
        print(f"Verified {args.output.relative_to(REPO_ROOT)}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
