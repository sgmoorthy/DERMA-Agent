"""
Targeted tests for research-paper-aligned enhancements.

Covers:
- Session-wide Benjamini-Hochberg FDR correction in the discovery engine
- Attention-style slide pooling in the WSI perception layer
"""

import os
import sys
import unittest

import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derma_agent.derma_core.agents.discovery_engine import (
    DiscoveryConfig,
    DiscoveryResult,
    FastDiscoveryEngine,
)
from derma_agent.derma_core.perception.wsi_engine import WSIEngine


class TestDiscoveryEngineFDR(unittest.TestCase):
    def _make_engine(self, p_values):
        engine = object.__new__(FastDiscoveryEngine)
        engine.config = DiscoveryConfig(significance_threshold=0.05)
        engine.results = []
        engine.ledger = []

        for idx, p_value in enumerate(p_values, start=1):
            raw_significant = (
                p_value is not None and p_value < engine.config.significance_threshold
            )
            result = DiscoveryResult(
                hypothesis=f"Hypothesis {idx}",
                test_code="print('ok')",
                execution_result="ok",
                conclusion="test",
                p_value=p_value,
                significant=raw_significant,
                raw_significant=raw_significant,
                fdr_alpha=engine.config.significance_threshold,
            )
            engine.results.append(result)
            engine.ledger.append({"project_id": "TCGA-SKCM", **result.to_dict()})
        return engine

    def test_benjamini_hochberg_correction_updates_significance(self):
        engine = self._make_engine([0.01, 0.04, 0.2])

        engine._apply_global_fdr_correction()

        adjusted = [result.adjusted_p_value for result in engine.results]
        self.assertIsNotNone(adjusted[0])
        self.assertIsNotNone(adjusted[1])
        self.assertIsNotNone(adjusted[2])
        self.assertAlmostEqual(float(adjusted[0]), 0.03, places=6)
        self.assertAlmostEqual(float(adjusted[1]), 0.06, places=6)
        self.assertAlmostEqual(float(adjusted[2]), 0.2, places=6)

        self.assertTrue(engine.results[0].significant)
        self.assertFalse(engine.results[1].significant)
        self.assertFalse(engine.results[2].significant)

        self.assertEqual(engine.ledger[0]["global_test_count"], 3)
        self.assertAlmostEqual(engine.ledger[1]["adjusted_p_value"], 0.06, places=6)

    def test_summary_report_exposes_fdr_metadata(self):
        engine = self._make_engine([0.001, 0.2])
        engine._apply_global_fdr_correction()

        report = engine.generate_summary_report()

        self.assertEqual(report["fdr_method"], "benjamini-hochberg")
        self.assertEqual(report["fdr_alpha"], 0.05)
        self.assertEqual(report["global_test_count"], 2)
        self.assertEqual(report["significant_findings"], 1)
        self.assertEqual(report["raw_significant_findings"], 1)
        self.assertIn("adjusted_p_value", report["top_findings"][0])


class TestWSIAttentionPooling(unittest.TestCase):
    def test_slide_representation_is_deterministic(self):
        engine = WSIEngine()

        rep_a = engine.get_slide_representation("WSI-TCGA-SKCM-001")
        rep_b = engine.get_slide_representation("WSI-TCGA-SKCM-001")

        np.testing.assert_allclose(rep_a["embedding"], rep_b["embedding"])
        np.testing.assert_allclose(
            rep_a["attention_weights"], rep_b["attention_weights"]
        )
        self.assertEqual(rep_a["n_tiles"], rep_b["n_tiles"])

    def test_attention_weights_form_valid_probability_distribution(self):
        engine = WSIEngine(n_tiles=32, embedding_dim=128)

        rep = engine.get_slide_representation("WSI-TCGA-BRCA-002")
        weights = rep["attention_weights"]

        self.assertEqual(rep["embedding"].shape, (128,))
        self.assertEqual(rep["patch_embeddings"].shape, (32, 128))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertGreater(rep["effective_tiles"], 0.0)
        self.assertGreater(rep["attention_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
