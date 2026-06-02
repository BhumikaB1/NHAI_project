"""
recognition/matcher.py
-----------------------
Phase 4 — Face Matching

Cosine similarity on L2-normalized embeddings simplifies to a dot product.
Threshold calibration:
  > 0.75 → too aggressive, causes false rejects in varied lighting
  > 0.60 → correct starting point for MobileFaceNet + ArcFace
  > 0.65 → tighten after testing on your dataset

Embedding store: JSON file (dev) — swap for SQLCipher in production.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


SIMILARITY_THRESHOLD = 0.60   # start here, tune up after benchmarking


@dataclass
class MatchResult:
    similarity: float
    matched: bool
    matched_user_id: Optional[str] = None

    def to_dict(self):
        return {
            "similarity":       round(self.similarity, 4),
            "matched":          self.matched,
            "matchedUserId":    self.matched_user_id,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D float32 vectors.
    If both are already L2-normalized (which FaceRecognizer ensures),
    this is equivalent to the dot product — O(n), very fast.

    Returns value in [-1, 1]. Identical faces -> ~0.95-1.0.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EmbeddingStore:
    """
    Simple JSON-backed embedding store for development.
    In production: replace with SQLCipher-encrypted SQLite.

    Schema:
        {
            "user_001": {
                "name": "Rahul Sharma",
                "embedding": [0.012, -0.034, ...]   // 512 floats
            },
            ...
        }
    """

    def __init__(self, store_path: str = "models/embeddings.json"):
        self._path  = Path(store_path)
        self._store: Dict[str, dict] = {}
        if self._path.exists() and self._path.stat().st_size > 0:
            self._load()

    def _load(self):
        with open(self._path, "r") as f:
            raw = json.load(f)
        # Convert lists back to numpy arrays
        for uid, data in raw.items():
            self._store[uid] = {
                "name":      data["name"],
                "embedding": np.array(data["embedding"], dtype=np.float32),
            }
        print(f"[EmbeddingStore] Loaded {len(self._store)} identities.")

    def save(self):
        """Persist store to disk (serializes numpy arrays to lists)."""
        serializable = {
            uid: {
                "name":      data["name"],
                "embedding": data["embedding"].tolist(),
            }
            for uid, data in self._store.items()
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(serializable, f, indent=2)

    def register(self, user_id: str, name: str, embedding: np.ndarray):
        """Register a new identity. Overwrites existing entry for same ID."""
        self._store[user_id] = {
            "name":      name,
            "embedding": embedding.astype(np.float32),
        }
        self.save()
        print(f"[EmbeddingStore] Registered: {user_id} ({name})")

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> MatchResult:
        """
        Compare query embedding against all stored identities.
        Returns the best match above threshold, or no-match.

        For N stored identities, this is O(N * 512) dot products.
        For up to ~10,000 users this is fast enough on-device (<5ms).
        Beyond that, use FAISS or approximate nearest-neighbour.
        """
        if not self._store:
            return MatchResult(similarity=0.0, matched=False)

        best_score = -1.0
        best_uid   = None

        for uid, data in self._store.items():
            score = cosine_similarity(query_embedding, data["embedding"])
            if score > best_score:
                best_score = score
                best_uid   = uid

        matched = best_score >= threshold

        return MatchResult(
            similarity=best_score,
            matched=matched,
            matched_user_id=best_uid if matched else None,
        )

    def __len__(self):
        return len(self._store)
