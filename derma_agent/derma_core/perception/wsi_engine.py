"""
Perception Layer: WSI processing & Foundation Model API interfaces.
Loads/mocks Whole Slide Image inputs and calls foundation model (APOLLO/TITAN) API hooks.

This module now includes a deterministic attention-style slide pooling routine inspired
by the research formalism in `data/DERMA_agent.pdf`:
    z_slide = sum_k a_k z_k
    a_k ∝ exp(w^T tanh(V z_k))
"""

import os
from typing import Any, Dict

import numpy as np


class WSIEngine:
    def __init__(
        self,
        slides_dir: str = "data/wsi_slides",
        n_tiles: int = 64,
        embedding_dim: int = 768,
    ):
        self.slides_dir = slides_dir
        self.n_tiles = n_tiles
        self.embedding_dim = embedding_dim
        os.makedirs(slides_dir, exist_ok=True)

    def _seed_from_slide(self, slide_id: str) -> int:
        """Create a deterministic seed from the slide id."""
        return int(sum(ord(c) for c in slide_id))

    def _rng(self, slide_id: str, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self._seed_from_slide(slide_id) + offset)

    def ingest_slide(self, slide_id: str) -> dict:
        """
        Simulate slide ingestion, extracting basic metadata and tissue patch targets.
        """
        return {
            "slide_id": slide_id,
            "dimensions": (102400, 80400),
            "microns_per_pixel": 0.25,
            "formats": ["svs", "tiff"],
            "status": "ingested",
            "tiles_targeted": self.n_tiles,
        }

    def get_patch_embeddings(
        self, slide_id: str, n_tiles: int | None = None
    ) -> np.ndarray:
        """
        Generate deterministic mock tile embeddings for a slide.
        Each row represents one patch embedding z_k.
        """
        tile_count = n_tiles or self.n_tiles
        rng = self._rng(slide_id, offset=17)

        base_signal = rng.normal(
            loc=0.0, scale=0.2, size=(tile_count, self.embedding_dim)
        )
        latent_axes = rng.normal(loc=0.0, scale=1.0, size=(3, self.embedding_dim))
        tile_scores = rng.normal(loc=0.0, scale=1.0, size=(tile_count, 3))
        embeddings = base_signal + tile_scores @ latent_axes / np.sqrt(
            3 * self.embedding_dim
        )
        return embeddings.astype(np.float32)

    def compute_attention_pooling(
        self, patch_embeddings: np.ndarray, slide_id: str
    ) -> Dict[str, Any]:
        """
        Compute a gated attention-style pooled embedding.

        Implements a deterministic mock analogue of:
            a_k ∝ exp(w^T tanh(V z_k))
            z_slide = Σ_k a_k z_k
        """
        if patch_embeddings.ndim != 2:
            raise ValueError(
                "patch_embeddings must be a 2D array of shape (tiles, embedding_dim)"
            )

        rng = self._rng(slide_id, offset=101)
        attention_hidden = min(128, max(16, patch_embeddings.shape[1] // 6))
        V = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(patch_embeddings.shape[1]),
            size=(attention_hidden, patch_embeddings.shape[1]),
        )
        w = rng.normal(
            loc=0.0, scale=1.0 / np.sqrt(attention_hidden), size=(attention_hidden,)
        )

        logits = np.tanh(patch_embeddings @ V.T) @ w
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.sum(weights)

        pooled_embedding = weights @ patch_embeddings
        effective_tiles = float(1.0 / np.sum(np.square(weights)))
        attention_entropy = float(-np.sum(weights * np.log(weights + 1e-12)))

        return {
            "embedding": pooled_embedding.astype(np.float32),
            "attention_weights": weights.astype(np.float32),
            "tile_logits": logits.astype(np.float32),
            "effective_tiles": effective_tiles,
            "attention_entropy": attention_entropy,
            "n_tiles": int(patch_embeddings.shape[0]),
        }

    def get_slide_representation(
        self, slide_id: str, n_tiles: int | None = None
    ) -> Dict[str, Any]:
        """Return the pooled slide representation plus attention metadata."""
        patch_embeddings = self.get_patch_embeddings(slide_id, n_tiles=n_tiles)
        pooled = self.compute_attention_pooling(patch_embeddings, slide_id)
        pooled["patch_embeddings"] = patch_embeddings
        pooled["slide_id"] = slide_id
        return pooled

    def get_apollo_embeddings(self, slide_id: str) -> np.ndarray:
        """
        Hook into APOLLO Foundation Model API.
        Returns a deterministic attention-pooled slide embedding vector representation.
        """
        return self.get_slide_representation(slide_id)["embedding"]

    def get_titan_classification(self, slide_id: str) -> dict:
        """
        Hook into TITAN Foundation Model API for diagnostic/pathology classification.
        Derives stable mock tissue metrics from the pooled slide representation.
        """
        slide_repr = self.get_slide_representation(slide_id)
        embedding = slide_repr["embedding"]
        attention_weights = slide_repr["attention_weights"]
        rng = self._rng(slide_id, offset=303)

        dominant_tile_fraction = float(np.max(attention_weights))
        nuclei_count = int(
            np.clip(1000 + 3500 * np.mean(np.abs(embedding[:64])), 1000, 5000)
        )
        cellularity = float(
            np.clip(0.2 + 0.55 * np.mean(np.abs(embedding[64:128])), 0.1, 0.85)
        )
        tissue_density = float(
            np.clip(0.25 + 0.5 * np.mean(np.abs(embedding[128:192])), 0.2, 0.9)
        )

        pattern_score = float(np.mean(embedding[192:256])) + dominant_tile_fraction
        if pattern_score > 0.65:
            primary_pattern = "infiltrative"
        elif pattern_score > 0.2:
            primary_pattern = "mixed"
        elif pattern_score > -0.15:
            primary_pattern = "nested"
        else:
            primary_pattern = "sheet-like"

        return {
            "slide_id": slide_id,
            "nuclei_count": nuclei_count,
            "cellularity": cellularity,
            "tissue_density": tissue_density,
            "primary_pattern": primary_pattern,
            "attention_entropy": float(slide_repr["attention_entropy"]),
            "effective_tiles": float(slide_repr["effective_tiles"]),
            "dominant_tile_fraction": dominant_tile_fraction,
            "embedding_checksum": float(np.mean(embedding[:16])),
            "mock_confidence": float(np.clip(0.55 + rng.uniform(0.0, 0.25), 0.0, 1.0)),
        }
