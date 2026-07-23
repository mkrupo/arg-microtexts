from __future__ import annotations

import unittest

from tools.build_edu_explorer_data import build_dataset


class EduExplorerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_dataset()

    def test_dataset_preserves_corpus_totals(self) -> None:
        self.assertEqual(len(self.dataset["documents"]), 112)
        self.assertEqual(
            self.dataset["summary"]["corpus"],
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

    def test_published_internal_proposal_counts(self) -> None:
        documents = self.dataset["documents"]
        self.assertEqual(sum(item["stats"]["edusegDocument"] for item in documents), 118)
        self.assertEqual(sum(item["stats"]["edusegAdu"] for item in documents), 121)
        self.assertEqual(sum(item["stats"]["secorel"] for item in documents), 133)

    def test_curated_coordination_case_uses_exact_offsets(self) -> None:
        document = next(
            item for item in self.dataset["documents"] if item["id"] == "micro_b001"
        )
        boundary = next(
            item for item in document["boundaries"] if item["offset"] == 405
        )
        self.assertFalse(boundary["gold"])
        self.assertEqual(boundary["aduId"], "a5")
        self.assertTrue(
            all(layer["predicted"] for layer in boundary["layers"].values())
        )
        self.assertEqual(
            document["german"]["text"][boundary["offset"] :],
            "und Vorreiter im Mülltrennen werden!",
        )

    def test_english_gold_boundaries_remain_separate_layer(self) -> None:
        self.assertEqual(
            sum(
                item["stats"]["englishInternalGold"]
                for item in self.dataset["documents"]
            ),
            104,
        )
        self.assertTrue(
            all(
                "layers" not in edu
                for document in self.dataset["documents"]
                for adu in document["english"]["adus"]
                for edu in adu["edus"]
            )
        )


if __name__ == "__main__":
    unittest.main()
