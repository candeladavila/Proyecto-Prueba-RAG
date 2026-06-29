import importlib.util
import shutil
import subprocess
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parent
DOCUMENTOS_DIR = BACKEND_DIR / "documentos"

RAG_SCRIPT_PATH = BACKEND_DIR / "06_ask_rag.py"
SCRIPT_PARSE_PATH = BACKEND_DIR / "01_liteparse_pdf_a_txt.py"
SCRIPT_CHUNKS_PATH = BACKEND_DIR / "02_langchain_chunking.py"
SCRIPT_QDRANT_PATH = BACKEND_DIR / "05_subir_chunks_a_qdrant.py"


def load_rag_module():
    """
    Carga 06_ask_rag.py aunque el nombre empiece por número.
    """
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    spec = importlib.util.spec_from_file_location("ask_rag_module", RAG_SCRIPT_PATH)

    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar 06_ask_rag.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


rag = load_rag_module()

app = FastAPI(
    title="Proyecto Prueba RAG API",
    description="API local para preguntar al RAG usando Qdrant + Gemini",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    filename: str
    page: str
    chunk_index: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    message: str
    filename: str
    saved_to: str
    parsed: bool
    chunked: bool


def sanitize_filename(filename: str) -> str:
    """
    Evita nombres peligrosos y mantiene solo caracteres seguros.
    """
    name = Path(filename).name
    safe_chars = []

    for char in name:
        if char.isalnum() or char in ("-", "_", ".", " "):
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_name = "".join(safe_chars).strip()

    if not safe_name:
        safe_name = "documento.pdf"

    return safe_name


def run_script(script_path: Path) -> str:
    """
    Ejecuta un script Python dentro de backend/.
    Usamos sys.executable para asegurar que usa la .venv activa.
    """
    if not script_path.exists():
        raise RuntimeError(f"No existe el script: {script_path.name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error ejecutando {script_path.name}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout

def progress_event(status: str, message: str, step: str | None = None) -> str:
    return json.dumps(
        {
            "status": status,
            "step": step,
            "message": message,
        },
        ensure_ascii=False,
    ) + "\n"

def process_uploaded_documents() -> None:
    """
    Ejecuta los pasos automáticos tras subir un PDF:

    Paso 1: PDF -> TXT + pages.jsonl
    Paso 2: pages.jsonl -> chunks.jsonl

    Si quieres que el documento quede disponible inmediatamente para el chat,
    puedes activar también el paso de Qdrant al final.
    """
    run_script(SCRIPT_PARSE_PATH)
    run_script(SCRIPT_CHUNKS_PATH)
    run_script(SCRIPT_QDRANT_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Backend RAG funcionando",
        "ask_endpoint": "POST /ask",
        "upload_endpoint": "POST /upload",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-stream")
async def upload_pdf_stream(file: UploadFile = File(...)):
    filename = sanitize_filename(file.filename or "documento.pdf")

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten archivos PDF.",
        )

    DOCUMENTOS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTOS_DIR / filename

    file_content = await file.read()
    await file.close()

    def event_generator():
        try:
            yield progress_event(
                status="progress",
                step="save",
                message=f"Guardando PDF: {filename}",
            )

            destination.write_bytes(file_content)

            yield progress_event(
                status="progress",
                step="parse",
                message="Extrayendo texto del PDF...",
            )

            run_script(SCRIPT_PARSE_PATH)

            yield progress_event(
                status="progress",
                step="chunks",
                message="Creando chunks con metadata...",
            )

            run_script(SCRIPT_CHUNKS_PATH)

            yield progress_event(
                status="progress",
                step="qdrant",
                message="Subiendo chunks a Qdrant...",
            )

            run_script(SCRIPT_QDRANT_PATH)

            yield progress_event(
                status="done",
                step="done",
                message=f'PDF "{filename}" procesado e indexado correctamente.',
            )

        except Exception as error:
            yield progress_event(
                status="error",
                step="error",
                message=f"Error procesando el PDF: {str(error)}",
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        rag.validate_config()

        chunks: list[dict[str, Any]] = rag.search_qdrant(
            question=question,
            top_k=request.top_k,
        )

        if not chunks:
            return AskResponse(
                answer="No he encontrado información relevante en los documentos.",
                sources=[],
            )

        context = rag.format_context(
            chunks=chunks,
            max_chars=rag.RAG_MAX_CONTEXT_CHARS,
        )

        prompt = rag.build_prompt(
            question=question,
            context=context,
        )

        answer = rag.ask_gemini(prompt)

        sources = []

        for chunk in chunks:
            sources.append(
                Source(
                    filename=str(chunk.get("filename", "desconocido")),
                    page=str(chunk.get("page_label", chunk.get("page", "desconocida"))),
                    chunk_index=str(chunk.get("chunk_index", "desconocido")),
                    score=float(chunk.get("score", 0.0)),
                )
            )

        return AskResponse(
            answer=answer or "Gemini no devolvió respuesta.",
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception as error:
        error_text = str(error)

        if "Gemini está saturado" in error_text:
            raise HTTPException(
                status_code=503,
                detail=error_text,
            )

        if "límite temporal de Gemini" in error_text:
            raise HTTPException(
                status_code=429,
                detail=error_text,
            )

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la pregunta: {error_text}",
        )