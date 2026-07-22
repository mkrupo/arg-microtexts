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
    check_outputs,
    expected_outputs,
    write_outputs,
)


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
            write_outputs(self.audit, edu_dir, boundary_path, manifest_path)
            check_outputs(self.audit, edu_dir, boundary_path, manifest_path)
            self.assertEqual(len(list(edu_dir.glob("*.edus"))), 112)
            self.assertEqual(len(json.loads(manifest_path.read_text())["documents"]), 112)

    def test_expected_outputs_are_stable(self) -> None:
        files, boundaries, manifest = expected_outputs(self.audit)
        self.assertEqual(len(files), 112)
        self.assertEqual(len(boundaries.splitlines()), 681)
        self.assertEqual(manifest["counts"], self.audit.totals)

    def test_committed_source_manifest_is_current(self) -> None:
        path = REPO_ROOT / "derived" / "source_audit.manifest.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.audit.manifest())

    def test_rst_punctuation_is_authoritative_for_known_variant(self) -> None:
        document = next(
            document for document in self.audit.english_documents if document.doc_id == "micro_d14"
        )
        self.assertEqual(document.edus[-1].text, "he was suddenly gone.")


if __name__ == "__main__":
    unittest.main()
