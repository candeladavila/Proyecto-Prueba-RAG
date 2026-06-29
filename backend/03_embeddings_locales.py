from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNKS_FILE, EMBEDDING_MODEL, EMBEDDINGS_DIR, EMBEDDINGS_FILE, METADATA_FILE


def main() -> None:
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    rows = [
        json.loads(line)
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [row["text"] for row in rows]

    if not texts:
        raise RuntimeError("No hay chunks. Ejecuta antes 02_langchain_chunking.py")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(EMBEDDINGS_FILE, embeddings)
    METADATA_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Modelo: {EMBEDDING_MODEL}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Guardado en: {EMBEDDINGS_FILE}")
    print(f"Metadatos: {METADATA_FILE}")


if __name__ == "__main__":
    main()
