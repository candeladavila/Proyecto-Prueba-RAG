from __future__ import annotations

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDINGS_FILE, METADATA_FILE


def buscar(query: str, top_k: int = 4) -> None:
    embeddings = np.load(EMBEDDINGS_FILE)
    rows = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Como normalizamos embeddings, producto escalar equivale a similitud coseno.
    scores = embeddings @ q_emb
    best_idx = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(best_idx, start=1):
        row = rows[int(idx)]
        metadata = row.get("metadata", {})
        source = metadata.get("source") or metadata.get("filename")
        page = metadata.get("page")

        print("=" * 80)
        print(
            f"#{rank} score={scores[idx]:.4f} "
            f"id={row.get('id')} source={source} page={page} "
            f"chunk={metadata.get('chunk_index')}"
        )
        print(row["text"][:1200])
        print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "que paso en 1926 con Mercedes"
    buscar(query)
