import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

from config import (
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


def validate_config() -> None:
    missing = []

    if not QDRANT_URL:
        missing.append("QDRANT_URL")

    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if not QDRANT_COLLECTION:
        missing.append("QDRANT_COLLECTION")

    if missing:
        raise RuntimeError("Faltan variables en .env: " + ", ".join(missing))


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_path}")

    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            text = item.get("text", "").strip()

            if text:
                chunks.append(item)

    return chunks


def get_embeddings_model() -> HuggingFaceEmbeddings:
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    print("Conectado a Qdrant Cloud.")

    return client


def get_existing_vector_size(client: QdrantClient, collection_name: str) -> int | None:
    info = client.get_collection(collection_name=collection_name)
    vectors_config = info.config.params.vectors

    if hasattr(vectors_config, "size"):
        return vectors_config.size

    if isinstance(vectors_config, dict):
        first_vector_config = next(iter(vectors_config.values()))
        return first_vector_config.size

    return None


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Crea la colección si no existe.
    Si existe, comprueba que la dimensión sea compatible.
    """
    if client.collection_exists(collection_name=collection_name):
        existing_size = get_existing_vector_size(client, collection_name)

        if existing_size != vector_size:
            raise RuntimeError(
                f"La colección {collection_name} existe con dimensión {existing_size}, "
                f"pero el modelo actual genera dimensión {vector_size}. "
                "Borra la colección o usa otra colección."
            )

        print(
            f"Colección existente válida: {collection_name} | dimensión={existing_size}"
        )
        return

    print(f"Creando colección: {collection_name} | dimensión={vector_size}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Colección creada: {collection_name}")


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Qdrant Cloud necesita índice de payload para filtrar por document_id.
    Este índice permite borrar chunks antiguos de un documento concreto.
    """
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

        print("Índice payload creado: document_id keyword")

    except Exception as error:
        error_text = str(error).lower()

        already_exists_messages = [
            "already exists",
            "already",
            "conflict",
            "same index",
        ]

        if any(message in error_text for message in already_exists_messages):
            print("Índice payload ya existe: document_id keyword")
            return

        raise


def extract_document_ids(chunks: list[dict[str, Any]]) -> set[str]:
    document_ids = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        document_id = metadata.get("document_id") or chunk.get("document_id")

        if document_id:
            document_ids.add(str(document_id))

    return document_ids


def delete_existing_documents(
    client: QdrantClient,
    collection_name: str,
    document_ids: set[str],
) -> None:
    """
    Borra de Qdrant solo los chunks antiguos de los documentos que se van a reindexar.
    No borra la colección completa.
    """
    for document_id in sorted(document_ids):
        print(f"Borrando chunks antiguos: document_id={document_id}")

        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )


def build_points(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> list[models.PointStruct]:
    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        document_id = metadata.get("document_id") or chunk.get("document_id") or "unknown"
        chunk_index = metadata.get("chunk_index", index)

        payload = {
            "text": text,
            **metadata,
        }

        payload["document_id"] = str(document_id)
        payload["chunk_index"] = chunk_index

        raw_id = chunk.get("id") or f"{document_id}_{chunk_index}_{index}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    return points


def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int = 64,
) -> None:
    total = len(points)

    for start in range(0, total, batch_size):
        end = start + batch_size
        batch = points[start:end]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

        print(f"Subidos points {start + 1}-{min(end, total)} de {total}")


def query_qdrant(
    client: QdrantClient,
    embeddings_model: HuggingFaceEmbeddings,
    question: str,
    top_k: int,
) -> None:
    print("\nProbando búsqueda en Qdrant...")
    print(f"Pregunta: {question}")

    query_vector = embeddings_model.embed_query(question)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    print("\nResultados encontrados:\n")

    for index, point in enumerate(results.points, start=1):
        payload = point.payload or {}

        print(f"Resultado {index}")
        print(f"Score: {point.score:.4f}")
        print(f"Archivo: {payload.get('filename', 'desconocido')}")
        print(f"Página: {payload.get('page_label', payload.get('page', 'desconocida'))}")
        print(f"Document ID: {payload.get('document_id', 'desconocido')}")
        print("Texto:")
        print((payload.get("text") or "")[:800])
        print("-" * 80)


def recreate_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(collection_name=collection_name):
        print(f"Borrando colección completa: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    print(f"Creando colección desde cero: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Colección recreada: {collection_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sube chunks a Qdrant y permite búsqueda de prueba."
    )

    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Borra toda la colección y la crea de nuevo.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Pregunta opcional para probar búsqueda en Qdrant después de subir.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de resultados para la query de prueba.",
    )

    args = parser.parse_args()

    validate_config()

    chunks_path = Path(CHUNKS_FILE)

    print(f"Leyendo chunks desde: {chunks_path}")

    chunks = load_chunks(chunks_path)
    print(f"Chunks con texto: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No hay chunks con texto para subir.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings_model = get_embeddings_model()

    print("Generando embeddings...")
    vectors = embeddings_model.embed_documents(texts)

    if not vectors:
        raise RuntimeError("No se han generado embeddings.")

    vector_size = len(vectors[0])
    print(f"Dimensión de embeddings: {vector_size}")

    client = get_qdrant_client()

    if args.recreate_collection:
        recreate_collection(
            client=client,
            collection_name=QDRANT_COLLECTION,
            vector_size=vector_size,
        )
    else:
        ensure_collection(
            client=client,
            collection_name=QDRANT_COLLECTION,
            vector_size=vector_size,
        )

    ensure_payload_indexes(
        client=client,
        collection_name=QDRANT_COLLECTION,
    )

    document_ids = extract_document_ids(chunks)

    if document_ids:
        delete_existing_documents(
            client=client,
            collection_name=QDRANT_COLLECTION,
            document_ids=document_ids,
        )
    else:
        print("No se han encontrado document_id en los chunks. No se borra nada antes del upsert.")

    points = build_points(
        chunks=chunks,
        vectors=vectors,
    )

    print(f"Subiendo {len(points)} chunks a Qdrant...")

    upload_points(
        client=client,
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print("Subida completada.")

    if args.query:
        query_qdrant(
            client=client,
            embeddings_model=embeddings_model,
            question=args.query,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()