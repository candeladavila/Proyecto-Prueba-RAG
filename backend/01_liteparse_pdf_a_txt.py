from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from liteparse import LiteParse
from pypdf import PdfReader

from config import DOCUMENTOS_DIR, PARSED_DIR, PAGES_SUFFIX

HEADER_RE = re.compile(
    r"Mercedes-Benz\s*-\s*documento de prueba RAG\s*\|\s*P[aá]gina\s*\d+",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HEADER_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def liteparse_full_text(pdf_path: Path) -> str:
    """Extrae texto completo con LiteParse. Lo usamos como TXT completo/fallback."""
    parser = LiteParse(
        output_format="text",
        ocr_enabled=False,
        quiet=True,
    )
    result: Any = parser.parse(str(pdf_path))
    return normalize_text(getattr(result, "text", str(result)))


def extract_pages_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    LiteParse devuelve un texto completo, pero para RAG necesitamos saber la página.
    Por eso extraemos página a página con pypdf y guardamos un JSONL intermedio.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue

        pages.append(
            {
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "source": pdf_path.name,
                "source_type": "pdf",
                "page": page_index,
                "page_label": str(page_index),
                "text": text,
            }
        )

    return pages


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    DOCUMENTOS_DIR.mkdir(exist_ok=True)
    PARSED_DIR.mkdir(exist_ok=True)

    pdfs = sorted(DOCUMENTOS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No hay PDFs en {DOCUMENTOS_DIR.resolve()}")
        return

    for pdf_path in pdfs:
        print(f"Parseando: {pdf_path.name}")

        # 1) TXT completo con LiteParse, útil para revisión manual.
        try:
            full_text = liteparse_full_text(pdf_path)
        except Exception as exc:
            print(f"Aviso: LiteParse falló con {pdf_path.name}: {exc}")
            full_text = ""

        # 2) JSONL página a página, necesario para metadata fiable en RAG.
        try:
            pages = extract_pages_with_pypdf(pdf_path)
        except Exception as exc:
            print(f"Aviso: extracción por páginas falló con {pdf_path.name}: {exc}")
            pages = []

        if not pages and full_text:
            # Fallback para no romper el flujo si pypdf no puede sacar páginas.
            pages = [
                {
                    "document_id": pdf_path.stem,
                    "filename": pdf_path.name,
                    "source": pdf_path.name,
                    "source_type": "pdf",
                    "page": None,
                    "page_label": None,
                    "text": full_text,
                }
            ]

        txt_path = PARSED_DIR / f"{pdf_path.stem}.txt"
        pages_path = PARSED_DIR / f"{pdf_path.stem}{PAGES_SUFFIX}"

        if not full_text and pages:
            full_text = "\n\n".join(page["text"] for page in pages)

        txt_path.write_text(full_text, encoding="utf-8")
        write_jsonl(pages_path, pages)

        print(f"OK TXT   -> {txt_path}")
        print(f"OK PAGES -> {pages_path} ({len(pages)} páginas con texto)")


if __name__ == "__main__":
    main()
