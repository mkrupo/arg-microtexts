from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus_data import REPO_ROOT, build_audit, normalized_join
from tools.export_english_edus import (
    boundary_tsv,
    check_outputs as check_english_outputs,
    expected_outputs as expected_english_outputs,
    write_outputs as write_english_outputs,
)
from tools.export_german_adus import (
    alignment_rows,
    boundary_rows as german_boundary_rows,
    check_outputs as check_german_outputs,
    expected_outputs as expected_german_outputs,
    write_outputs as write_german_outputs,
)
from tools.publish_eduseg_run import check_published as check_published_eduseg


class CorpusPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = build_audit()

    def test_expected_totals(self) -> None:
        self.assertEqual(
            self.audit.totals,
            {
                "documents": 112,
                "adus": 576,
                "english_edus": 680,
                "split_adus": 83,
                "internal_boundaries": 104,
                "refined_documents": 53,
                "sameunit_documents": 9,
            },
        )

    def test_english_edus_reconstruct_raw_text(self) -> None:
        for document in self.audit.english_documents:
            self.assertEqual(normalized_join(span.text for span in document.edus), document.raw_text)

    def test_german_adus_reconstruct_raw_text(self) -> None:
        for document in self.audit.german_documents:
            self.assertEqual(normalized_join(span.text for span in document.adus), document.raw_text)

    def test_bilingual_adu_alignment(self) -> None:
        rows = alignment_rows(self.audit)
        self.assertEqual(len(rows), 576)
        self.assertTrue(all(row["de_text"] and row["en_text"] for row in rows))
        self.assertEqual(
            [(row["doc_id"], row["adu_id"]) for row in rows],
            [
                (document.doc_id, span.adu_id)
                for document in self.audit.german_documents
                for span in document.adus
            ],
        )

    def test_boundary_classes(self) -> None:
        rows = list(csv.DictReader(io.StringIO(boundary_tsv(self.audit)), delimiter="\t"))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["boundary_class"]] = counts.get(row["boundary_class"], 0) + 1
        self.assertEqual(
            counts,
            {"document_start": 112, "adu": 464, "internal_edu": 104},
        )

    def test_export_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            edu_dir = root / "edus"
            boundary_path = root / "boundaries.tsv"
            manifest_path = root / "manifest.json"
            write_english_outputs(self.audit, edu_dir, boundary_path, manifest_path)
            check_english_outputs(self.audit, edu_dir, boundary_path, manifest_path)
            self.assertEqual(len(list(edu_dir.glob("*.edus"))), 112)
            self.assertEqual(len(json.loads(manifest_path.read_text())["documents"]), 112)

    def test_expected_outputs_are_stable(self) -> None:
        files, boundaries, manifest = expected_english_outputs(self.audit)
        self.assertEqual(len(files), 112)
        self.assertEqual(len(boundaries.splitlines()), 681)
        self.assertEqual(manifest["counts"], self.audit.totals)

    def test_german_boundary_classes(self) -> None:
        rows = german_boundary_rows(self.audit)
        counts: dict[str, int] = {}
        for row in rows:
            boundary_class = str(row["boundary_class"])
            counts[boundary_class] = counts.get(boundary_class, 0) + 1
        self.assertEqual(counts, {"document_start": 112, "adu": 464})

    def test_german_export_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adu_dir = root / "adus"
            boundary_path = root / "boundaries.tsv"
            alignment_path = root / "alignments.tsv"
            manifest_path = root / "manifest.json"
            write_german_outputs(
                self.audit, adu_dir, boundary_path, alignment_path, manifest_path
            )
            check_german_outputs(
                self.audit, adu_dir, boundary_path, alignment_path, manifest_path
            )
            self.assertEqual(len(list(adu_dir.glob("*.adus"))), 112)
            _, boundaries, alignments, manifest = expected_german_outputs(self.audit)
            self.assertEqual(len(boundaries.splitlines()), 577)
            self.assertEqual(len(alignments.splitlines()), 577)
            self.assertEqual(manifest["counts"]["adus"], 576)

    def test_committed_source_manifest_is_current(self) -> None:
        path = REPO_ROOT / "derived" / "source_audit.manifest.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.audit.manifest())

    def test_rst_punctuation_is_authoritative_for_known_variant(self) -> None:
        document = next(
            document for document in self.audit.english_documents if document.doc_id == "micro_d14"
        )
        self.assertEqual(document.edus[-1].text, "he was suddenly gone.")

    def test_published_eduseg_layer_is_current(self) -> None:
        check_published_eduseg()


if __name__ == "__main__":
    unittest.main()
