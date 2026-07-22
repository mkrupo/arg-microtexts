"""Shared, dependency-free corpus parsing and validation helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MULTILAYER_ROOT = REPO_ROOT / "external" / "arg-microtexts-multilayer"
EXPECTED_DOCUMENTS = 112
EXPECTED_ADUS = 576
EXPECTED_ENGLISH_EDUS = 680
EXPECTED_SPLIT_ADUS = 83
EXPECTED_INTERNAL_BOUNDARIES = 104
EXPECTED_REFINED_DOCUMENTS = 53
EXPECTED_SAMEUNIT_DOCUMENTS = 9
KNOWN_RST_ARGUMENT_TEXT_VARIANTS = {
    ("micro_d14", "e7"): ("he was suddenly gone", "he was suddenly gone."),
}
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


class AuditError(RuntimeError):
    """Raised when a corpus invariant does not hold."""


@dataclass(frozen=True)
class UnitSpan:
    unit_id: str
    text: str
    start: int
    end: int
    adu_id: str | None = None


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    german_raw_sha256: str
    english_raw_sha256: str
    german_adu_count: int
    english_adu_count: int
    english_edu_count: int
    split_adu_count: int
    internal_boundary_count: int
    sameunit_affected: bool


@dataclass(frozen=True)
class EnglishDocument:
    doc_id: str
    raw_text: str
    original_adus: tuple[UnitSpan, ...]
    edus: tuple[UnitSpan, ...]
    sameunit_affected: bool


@dataclass(frozen=True)
class GermanDocument:
    doc_id: str
    raw_text: str
    adus: tuple[UnitSpan, ...]
    sameunit_affected: bool


@dataclass(frozen=True)
class CorpusAudit:
    original_corpus_tree: str
    source_commit: str
    documents: tuple[DocumentRecord, ...]
    german_documents: tuple[GermanDocument, ...]
    english_documents: tuple[EnglishDocument, ...]

    @property
    def totals(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "adus": sum(document.english_adu_count for document in self.documents),
            "english_edus": sum(document.english_edu_count for document in self.documents),
            "split_adus": sum(document.split_adu_count for document in self.documents),
            "internal_boundaries": sum(
                document.internal_boundary_count for document in self.documents
            ),
            "refined_documents": sum(
                document.internal_boundary_count > 0 for document in self.documents
            ),
            "sameunit_documents": sum(document.sameunit_affected for document in self.documents),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "original_corpus": {
                "path": "corpus",
                "tree_sha": self.original_corpus_tree,
            },
            "multilayer_source": {
                "path": "external/arg-microtexts-multilayer",
                "url": "https://github.com/peldszus/arg-microtexts-multilayer.git",
                "commit": self.source_commit,
            },
            "whitespace_policy": (
                "Original corpus .txt is authoritative; surrounding source-unit whitespace "
                "and trailing multilayer whitespace are ignored; internal text is unchanged."
            ),
            "known_source_variants": [
                {
                    "doc_id": doc_id,
                    "argument_edu_id": edu_id,
                    "argument_text": texts[0],
                    "rst_text": texts[1],
                    "resolution": "Use the RST text; it preserves the authoritative raw text.",
                }
                for (doc_id, edu_id), texts in sorted(KNOWN_RST_ARGUMENT_TEXT_VARIANTS.items())
            ],
            "totals": self.totals,
            "documents": [asdict(document) for document in self.documents],
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_join(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def read_authoritative_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip()


def source_commit() -> str:
    if not (MULTILAYER_ROOT / ".git").exists():
        raise AuditError(
            "Multilayer submodule is not initialized. Run: "
            "git submodule update --init --recursive"
        )
    result = subprocess.run(
        ["git", "-C", str(MULTILAYER_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def original_corpus_tree() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:corpus"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def xml_edu_pairs(path: Path) -> list[tuple[str, str]]:
    root = ET.parse(path).getroot()
    pairs = []
    for edu in root.findall("edu"):
        text = (edu.text or "").strip()
        if not text:
            raise AuditError(f"Empty EDU {edu.attrib.get('id')} in {path}")
        pairs.append((edu.attrib["id"], text))
    return pairs


def locate_units(
    raw_text: str, units: Iterable[tuple[str, str]], *, label: str = "document"
) -> tuple[UnitSpan, ...]:
    spans: list[UnitSpan] = []
    cursor = 0
    for unit_id, text in units:
        while cursor < len(raw_text) and raw_text[cursor].isspace():
            cursor += 1
        if not raw_text.startswith(text, cursor):
            context = raw_text[cursor : cursor + max(80, len(text))]
            raise AuditError(
                f"Could not locate {label}/{unit_id} at character {cursor}. "
                f"Expected {text!r}; found {context!r}"
            )
        end = cursor + len(text)
        spans.append(UnitSpan(unit_id=unit_id, text=text, start=cursor, end=end))
        cursor = end
    if raw_text[cursor:].strip():
        raise AuditError(
            f"Uncovered text after final unit in {label} at character {cursor}: "
            f"{raw_text[cursor:]!r}"
        )
    return tuple(spans)


def segmentation_targets(path: Path, edu_ids: Iterable[str]) -> dict[str, str]:
    root = ET.parse(path).getroot()
    adu_ids = {adu.attrib["id"] for adu in root.findall("adu")}
    targets: dict[str, str] = {}
    for edge in root.findall("edge"):
        if edge.attrib.get("type") == "seg":
            targets[edge.attrib["src"]] = edge.attrib["trg"]

    mapping: dict[str, str] = {}
    for edu_id in edu_ids:
        current = edu_id
        visited = {current}
        while current not in adu_ids:
            if current not in targets:
                raise AuditError(f"No segmentation path from {edu_id} to an ADU in {path}")
            current = targets[current]
            if current in visited:
                raise AuditError(f"Segmentation cycle from {edu_id} in {path}")
            visited.add(current)
        mapping[edu_id] = current
    return mapping


def with_adu_ids(spans: Iterable[UnitSpan], mapping: dict[str, str]) -> tuple[UnitSpan, ...]:
    return tuple(
        UnitSpan(
            unit_id=span.unit_id,
            text=span.text,
            start=span.start,
            end=span.end,
            adu_id=mapping[span.unit_id],
        )
        for span in spans
    )


def rst_edu_pairs(path: Path) -> list[tuple[str, str]]:
    root = ET.parse(path).getroot()
    return [
        (segment.attrib["id"], (segment.text or "").strip())
        for segment in root.findall("./body/segment")
    ]


def assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise AuditError(f"{message}: expected {expected!r}, found {actual!r}")


def _document_ids(directory: Path, suffix: str) -> set[str]:
    return {path.stem for path in directory.glob(f"*{suffix}")}


def build_audit() -> CorpusAudit:
    if not MULTILAYER_ROOT.exists():
        raise AuditError(
            "Missing multilayer submodule. Run: git submodule update --init --recursive"
        )

    de_ids = _document_ids(REPO_ROOT / "corpus" / "de", ".xml")
    en_ids = _document_ids(REPO_ROOT / "corpus" / "en", ".xml")
    arg_ids = _document_ids(MULTILAYER_ROOT / "corpus" / "arg", ".xml")
    rst_ids = _document_ids(MULTILAYER_ROOT / "corpus" / "rst", ".rs3")
    assert_equal(len(de_ids), EXPECTED_DOCUMENTS, "German document count")
    assert_equal(en_ids, de_ids, "Original English/German document IDs")
    assert_equal(arg_ids, de_ids, "Multilayer argument/German document IDs")
    assert_equal(rst_ids, de_ids, "Multilayer RST/German document IDs")

    sameunit_ids = _document_ids(
        MULTILAYER_ROOT / "corpus" / "rst-without-sameunit", ".rs3"
    )
    assert_equal(len(sameunit_ids), EXPECTED_SAMEUNIT_DOCUMENTS, "Same-Unit document count")

    records: list[DocumentRecord] = []
    german_documents: list[GermanDocument] = []
    english_documents: list[EnglishDocument] = []

    for doc_id in sorted(de_ids):
        de_raw = read_authoritative_text(REPO_ROOT / "corpus" / "de" / f"{doc_id}.txt")
        en_raw = read_authoritative_text(REPO_ROOT / "corpus" / "en" / f"{doc_id}.txt")
        multilayer_raw = read_authoritative_text(
            MULTILAYER_ROOT / "corpus" / "txt" / f"{doc_id}.txt"
        )
        assert_equal(multilayer_raw, en_raw, f"Multilayer/original raw English for {doc_id}")

        de_xml = REPO_ROOT / "corpus" / "de" / f"{doc_id}.xml"
        en_xml = REPO_ROOT / "corpus" / "en" / f"{doc_id}.xml"
        fine_xml = MULTILAYER_ROOT / "corpus" / "arg" / f"{doc_id}.xml"

        de_pairs = xml_edu_pairs(de_xml)
        en_pairs = xml_edu_pairs(en_xml)
        argument_fine_pairs = xml_edu_pairs(fine_xml)
        rst_pairs = rst_edu_pairs(MULTILAYER_ROOT / "corpus" / "rst" / f"{doc_id}.rs3")
        de_spans = locate_units(de_raw, de_pairs, label=f"{doc_id}/de-original")
        en_spans = locate_units(en_raw, en_pairs, label=f"{doc_id}/en-original")
        rst_spans = locate_units(en_raw, rst_pairs, label=f"{doc_id}/en-rst")

        assert_equal(len(de_spans), len(en_spans), f"German/English ADU count for {doc_id}")
        de_mapping = segmentation_targets(de_xml, (span.unit_id for span in de_spans))
        en_mapping = segmentation_targets(en_xml, (span.unit_id for span in en_spans))
        fine_mapping = segmentation_targets(
            fine_xml, (edu_id for edu_id, _ in argument_fine_pairs)
        )
        de_adus = with_adu_ids(de_spans, de_mapping)
        en_adus = with_adu_ids(en_spans, en_mapping)
        assert_equal(
            len(rst_spans),
            len(argument_fine_pairs),
            f"RST/refined-argument EDU count for {doc_id}",
        )
        fine_spans = tuple(
            UnitSpan(
                unit_id=rst_span.unit_id,
                text=rst_span.text,
                start=rst_span.start,
                end=rst_span.end,
                adu_id=fine_mapping[argument_pair[0]],
            )
            for rst_span, argument_pair in zip(rst_spans, argument_fine_pairs)
        )
        for rst_span, (argument_id, argument_text) in zip(rst_spans, argument_fine_pairs):
            if rst_span.text == argument_text:
                continue
            expected_variant = KNOWN_RST_ARGUMENT_TEXT_VARIANTS.get((doc_id, argument_id))
            assert_equal(
                (argument_text, rst_span.text),
                expected_variant,
                f"RST/refined-argument text for {doc_id}/{argument_id}",
            )

        assert_equal(
            [span.adu_id for span in de_adus],
            [span.adu_id for span in en_adus],
            f"German/English ADU IDs for {doc_id}",
        )

        grouped: dict[str, list[UnitSpan]] = {}
        for span in fine_spans:
            grouped.setdefault(span.adu_id or "", []).append(span)
        assert_equal(set(grouped), {span.adu_id for span in en_adus}, f"Fine ADU coverage for {doc_id}")
        for adu in en_adus:
            assert_equal(
                normalized_join(span.text for span in grouped[adu.adu_id or ""]),
                adu.text,
                f"Fine EDU reconstruction of {doc_id}/{adu.adu_id}",
            )

        fine_starts = {span.start for span in fine_spans}
        original_starts = {span.start for span in en_adus}
        if not original_starts <= fine_starts:
            missing = sorted(original_starts - fine_starts)
            raise AuditError(f"ADU starts missing from fine EDUs in {doc_id}: {missing}")

        split_adu_count = sum(len(spans) > 1 for spans in grouped.values())
        internal_boundary_count = sum(len(spans) - 1 for spans in grouped.values())
        sameunit_affected = doc_id in sameunit_ids
        records.append(
            DocumentRecord(
                doc_id=doc_id,
                german_raw_sha256=sha256_text(de_raw),
                english_raw_sha256=sha256_text(en_raw),
                german_adu_count=len(de_adus),
                english_adu_count=len(en_adus),
                english_edu_count=len(fine_spans),
                split_adu_count=split_adu_count,
                internal_boundary_count=internal_boundary_count,
                sameunit_affected=sameunit_affected,
            )
        )
        german_documents.append(
            GermanDocument(
                doc_id=doc_id,
                raw_text=de_raw,
                adus=de_adus,
                sameunit_affected=sameunit_affected,
            )
        )
        english_documents.append(
            EnglishDocument(
                doc_id=doc_id,
                raw_text=en_raw,
                original_adus=en_adus,
                edus=fine_spans,
                sameunit_affected=sameunit_affected,
            )
        )

    audit = CorpusAudit(
        original_corpus_tree=original_corpus_tree(),
        source_commit=source_commit(),
        documents=tuple(records),
        german_documents=tuple(german_documents),
        english_documents=tuple(english_documents),
    )
    expected_totals = {
        "documents": EXPECTED_DOCUMENTS,
        "adus": EXPECTED_ADUS,
        "english_edus": EXPECTED_ENGLISH_EDUS,
        "split_adus": EXPECTED_SPLIT_ADUS,
        "internal_boundaries": EXPECTED_INTERNAL_BOUNDARIES,
        "refined_documents": EXPECTED_REFINED_DOCUMENTS,
        "sameunit_documents": EXPECTED_SAMEUNIT_DOCUMENTS,
    }
    assert_equal(audit.totals, expected_totals, "Corpus totals")
    return audit


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
