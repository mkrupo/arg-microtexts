from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.corpus_data import AuditError, REPO_ROOT, UnitSpan
from tools.run_eduseg_de import (
    CandidateScore,
    DocumentPrediction,
    boundary_rows,
    is_text_boundary,
    output_hashes,
    repository_path,
    segment_text,
    sha256_file,
    verify_model,
)


class EduSegRunnerTests(unittest.TestCase):
    def test_segment_text_round_trip(self) -> None:
        text = "Ein Argument, weil es relevant ist. Danach folgt ein zweites."
        self.assertEqual(
            segment_text(text, {0, 14, 36}),
            ["Ein Argument,", "weil es relevant ist.", "Danach folgt ein zweites."],
        )

    def test_segment_text_rejects_subword_boundary(self) -> None:
        self.assertFalse(is_text_boundary("Argument", 3))
        with self.assertRaises(AuditError):
            segment_text("Argument", {0, 3})

    def test_model_hash_contract_covers_all_tokenizer_files(self) -> None:
        names = {
            "config_json": "config.json",
            "model_safetensors": "model.safetensors",
            "tokenizer_json": "tokenizer.json",
            "sentencepiece_bpe_model": "sentencepiece.bpe.model",
            "special_tokens_map_json": "special_tokens_map.json",
            "tokenizer_config_json": "tokenizer_config.json",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            for filename in names.values():
                (model_dir / filename).write_text(filename, encoding="utf-8")
            config = {
                "model": {
                    "sha256": {
                        key: sha256_file(model_dir / filename)
                        for key, filename in names.items()
                    }
                }
            }
            self.assertEqual(verify_model(model_dir, config), config["model"]["sha256"])
            (model_dir / "tokenizer_config.json").write_text("changed", encoding="utf-8")
            with self.assertRaises(AuditError):
                verify_model(model_dir, config)

    def test_manifest_paths_and_output_hashes_are_portable(self) -> None:
        config = REPO_ROOT / "experiments" / "configs" / "eduseg_de_document_v1.toml"
        self.assertEqual(
            repository_path(config), "experiments/configs/eduseg_de_document_v1.toml"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "result.txt").write_text("result\n", encoding="utf-8")
            (root / "manifest.json").write_text("ignored\n", encoding="utf-8")
            self.assertEqual(
                output_hashes(root),
                [{"path": "result.txt", "sha256": sha256_file(root / "result.txt")}],
            )
            with self.assertRaises(AuditError):
                repository_path(root / "external.toml")

    def test_boundary_confidence_uses_strongest_piece_at_shared_offset(self) -> None:
        text = "Er sagt, dass es stimmt."
        document = SimpleNamespace(
            doc_id="micro_test",
            raw_text=text,
            adus=(UnitSpan("e1", text, 0, len(text), "a1"),),
            sameunit_affected=False,
        )
        prediction = DocumentPrediction(
            doc_id=document.doc_id,
            text=text,
            scores=(
                CandidateScore(4, 9, 10, "▁", 0.78, True),
                CandidateScore(5, 9, 13, "dass", 0.23, False),
            ),
            predicted_starts=frozenset({0, 9}),
            token_count=8,
            invalid_boundary_offsets=(),
        )
        rows = boundary_rows(
            audit=SimpleNamespace(german_documents=(document,)),
            predictions=[prediction],
            config={
                "run_id": "test",
                "model": {"name": "eduseg_de", "revision": "test"},
            },
            constrained=False,
        )
        internal = next(row for row in rows if row["boundary_class"] == "internal_edu")
        self.assertEqual(internal["confidence"], "0.78000000")


if __name__ == "__main__":
    unittest.main()
