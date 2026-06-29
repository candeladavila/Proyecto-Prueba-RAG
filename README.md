# Proyecto Prueba RAG

Proyecto RAG local con:

```text
React + Vite
FastAPI
LangChain
Embeddings locales con sentence-transformers
Qdrant Cloud
Gemini API
```

La aplicación permite:

```text
1. Subir documentos PDF desde el frontend
2. Guardarlos en backend/documentos/
3. Extraer texto del PDF
4. Extraer texto página a página
5. Crear chunks con metadata
6. Subir los chunks a Qdrant
7. Preguntar desde un chat web
8. Obtener respuestas generadas con Gemini usando los chunks recuperados
```

---

## Estructura del proyecto

```text
Proyecto-Prueba-RAG/
├── .venv/
├── backend/
│   ├── documentos/
│   ├── documentos-parseados/
│   ├── chunks/
│   ├── embeddings/
│   ├── 01_liteparse_pdf_a_txt.py
│   ├── 02_langchain_chunking.py
│   ├── 03_embeddings_locales.py
│   ├── 04_busqueda_semantica.py
│   ├── 05_subir_chunks_a_qdrant.py
│   ├── 06_ask_rag.py
│   ├── config.py
│   ├── main.py
│   ├── requirements.txt
│   └── .env
│
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── index.css
    │   └── main.jsx
    ├── .env.local
    ├── package.json
    └── vite.config.js
```

---

## Flujo completo del RAG

```text
Usuario sube PDF
↓
Frontend React llama a POST /upload-stream
↓
Backend guarda el PDF en backend/documentos/
↓
01_liteparse_pdf_a_txt.py extrae texto y páginas
↓
02_langchain_chunking.py crea chunks con metadata
↓
05_subir_chunks_a_qdrant.py crea embeddings y sube chunks a Qdrant
↓
Usuario pregunta en el chat
↓
Frontend llama a POST /ask
↓
Backend crea embedding de la pregunta
↓
Qdrant devuelve chunks relevantes
↓
Gemini genera respuesta final usando esos chunks
↓
Frontend muestra respuesta y fuentes
```

---

## 1. Crear entorno virtual

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
```

Activar entorno virtual:

```bash
source .venv/bin/activate
```

En Windows:

```bash
.venv\Scripts\activate
```

---

## 2. Instalar dependencias del backend

Desde la raíz:

```bash
cd backend
source ../.venv/bin/activate
pip install -r requirements.txt
```

Si falta alguna dependencia, instala:

```bash
pip install fastapi "uvicorn[standard]" python-multipart python-dotenv qdrant-client langchain-text-splitters langchain-huggingface sentence-transformers google-genai pypdf numpy
```

El archivo `backend/requirements.txt` debería incluir:

```txt
liteparse
langchain-text-splitters
langchain-huggingface
sentence-transformers
numpy
qdrant-client
python-dotenv
pypdf
google-genai
fastapi
uvicorn[standard]
python-multipart
```

---

## 3. Configurar variables de entorno del backend

Crear archivo:

```text
backend/.env
```

Contenido recomendado:

```env
QDRANT_URL=https://TU_CLUSTER.qdrant.io
QDRANT_API_KEY=TU_API_KEY_QDRANT
QDRANT_COLLECTION=documentos_rag

GEMINI_API_KEY=TU_API_KEY_GEMINI
GEMINI_MODEL=gemini-2.5-flash-lite

RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=5000

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Importante:

```text
No subir .env a GitHub.
No poner claves privadas en el frontend.
```

---

## 4. Configurar frontend

Desde la raíz:

```bash
cd frontend
npm install
```

Crear archivo:

```text
frontend/.env.local
```

Contenido:

```env
VITE_API_URL=http://127.0.0.1:8000
```

---

## 5. Arrancar backend

Desde la raíz:

```bash
cd backend
source ../.venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend disponible en:

```text
http://127.0.0.1:8000
```

Comprobar salud del backend:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{"status":"ok"}
```

---

## 6. Arrancar frontend

En otra terminal, desde la raíz:

```bash
cd frontend
npm run dev
```

Frontend disponible normalmente en:

```text
http://localhost:5173
```

---

## 7. Endpoints del backend

### `GET /health`

Comprueba que el backend está vivo.

```bash
curl http://127.0.0.1:8000/health
```

---

### `POST /upload-stream`

Sube un PDF, lo guarda en `backend/documentos/`, lo procesa y lo sube a Qdrant.

Este endpoint devuelve progreso paso a paso en formato NDJSON.

Flujo interno:

```text
Guardar PDF
↓
Extraer texto
↓
Crear chunks
↓
Subir chunks a Qdrant
↓
Finalizar
```

Prueba con `curl`:

```bash
curl -N -X POST "http://127.0.0.1:8000/upload-stream" \
  -F "file=@/ruta/a/tu/documento.pdf"
```

Ejemplo de respuesta:

```json
{"status":"progress","step":"save","message":"Guardando PDF: documento.pdf"}
{"status":"progress","step":"parse","message":"Extrayendo texto del PDF..."}
{"status":"progress","step":"chunks","message":"Creando chunks con metadata..."}
{"status":"progress","step":"qdrant","message":"Subiendo chunks a Qdrant..."}
{"status":"done","step":"done","message":"PDF \"documento.pdf\" procesado e indexado correctamente."}
```

---

### `POST /ask`

Recibe una pregunta y devuelve respuesta generada con Gemini usando Qdrant como fuente documental.

Request:

```json
{
  "question": "cuando se fundó Mercedes-Benz",
  "top_k": 3
}
```

Prueba con `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "cuando se fundó Mercedes-Benz", "top_k": 3}'
```

Respuesta esperada:

```json
{
  "answer": "Respuesta generada por Gemini...",
  "sources": [
    {
      "filename": "documento.pdf",
      "page": "1",
      "chunk_index": "0",
      "score": 0.78
    }
  ]
}
```

---

## 8. Subir PDFs desde el frontend

En la interfaz web:

```text
1. Pulsar “Elegir PDF”
2. Seleccionar un archivo .pdf
3. Pulsar “Subir”
4. Esperar a que el chat muestre:
   - Guardando PDF
   - Extrayendo texto
   - Creando chunks
   - Subiendo chunks a Qdrant
   - PDF procesado correctamente
5. Preguntar sobre el documento en el chat
```

Los PDFs se guardan en:

```text
backend/documentos/
```

---

## 9. Scripts del backend

### `01_liteparse_pdf_a_txt.py`

Convierte PDFs en texto y genera extracción página a página.

Entrada:

```text
backend/documentos/*.pdf
```

Salida:

```text
backend/documentos-parseados/*.txt
backend/documentos-parseados/*.pages.jsonl
```

Ejecutar manualmente:

```bash
cd backend
source ../.venv/bin/activate
python 01_liteparse_pdf_a_txt.py
```

---

### `02_langchain_chunking.py`

Lee los archivos `.pages.jsonl` y genera chunks con metadata.

Entrada:

```text
backend/documentos-parseados/*.pages.jsonl
```

Salida:

```text
backend/chunks/chunks.jsonl
```

Ejecutar manualmente:

```bash
python 02_langchain_chunking.py
```

---

### `03_embeddings_locales.py`

Genera embeddings locales a partir de los chunks.

Entrada:

```text
backend/chunks/chunks.jsonl
```

Salida:

```text
backend/embeddings/embeddings.npy
```

Ejecutar manualmente:

```bash
python 03_embeddings_locales.py
```

---

### `04_busqueda_semantica.py`

Permite probar búsqueda semántica local sin Qdrant.

Ejecutar:

```bash
python 04_busqueda_semantica.py
```

---

### `05_subir_chunks_a_qdrant.py`

Lee `chunks/chunks.jsonl`, genera embeddings y sube los chunks a Qdrant.

También:

```text
- Comprueba que la colección existe
- Valida la dimensión vectorial
- Crea el índice payload document_id si hace falta
- Borra chunks antiguos del mismo document_id
- Sube los chunks nuevos
```

Ejecutar:

```bash
python 05_subir_chunks_a_qdrant.py
```

Probar búsqueda después de subir:

```bash
python 05_subir_chunks_a_qdrant.py --query "cuando se fundó Mercedes-Benz" --top-k 5
```

---

### `06_ask_rag.py`

Realiza el flujo RAG completo en modo script:

```text
Pregunta
↓
Embedding local de la pregunta
↓
Búsqueda en Qdrant
↓
Construcción de contexto
↓
Respuesta con Gemini
```

Ejecutar:

```bash
python 06_ask_rag.py "cuando se fundó Mercedes-Benz" --show-sources
```

---

## 10. Rehacer completamente la colección de Qdrant

Si quieres borrar la colección completa y recrearla desde cero:

```bash
cd backend
source ../.venv/bin/activate
python 05_subir_chunks_a_qdrant.py --recreate-collection
```

También puedes recrearla y hacer una búsqueda de prueba:

```bash
python 05_subir_chunks_a_qdrant.py --recreate-collection --query "cuando se fundó Mercedes-Benz" --top-k 5
```

Usar este comando cuando:

```text
- Cambias el modelo de embeddings
- Cambias la dimensión vectorial
- La colección quedó inconsistente
- Quieres borrar datos antiguos
- Quieres preparar una demo desde cero
```

---

## 11. Limpieza parcial de Qdrant

El flujo normal no borra toda la colección.

Cuando subes un documento con el mismo `document_id`, el script:

```text
1. Busca chunks antiguos con ese document_id
2. Los borra
3. Sube los chunks nuevos
```

Para que esto funcione en Qdrant Cloud, el script crea un índice de payload:

```text
document_id → keyword
```

Esto evita errores como:

```text
Index required but not found for "document_id" of type keyword
```

---

## 12. Metadata de los chunks

Cada chunk se guarda con metadata útil para trazabilidad.

Ejemplo:

```json
{
  "id": "bmw_informacion_rag_p001_c000",
  "text": "Texto del chunk...",
  "metadata": {
    "source": "bmw_informacion_rag.pdf",
    "filename": "bmw_informacion_rag.pdf",
    "document_id": "bmw_informacion_rag",
    "source_type": "pdf",
    "page": 1,
    "page_label": "1",
    "language": "es",
    "chunk_index": 0,
    "chunk_index_in_page": 0,
    "chunk_size": 921
  }
}
```

En Qdrant, el payload queda con campos como:

```text
text
source
filename
document_id
source_type
page
page_label
language
chunk_index
chunk_index_in_page
chunk_size
```

Esto permite mostrar fuentes en la respuesta:

```text
bmw_informacion_rag.pdf, página 3
```

---

## 13. Configuración recomendada para Gemini

En `backend/.env`:

```env
GEMINI_MODEL=gemini-2.5-flash-lite
RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=5000
```

Si Gemini devuelve error 503:

```text
503 UNAVAILABLE
This model is currently experiencing high demand
```

No significa que el proyecto esté mal. Significa que Gemini está saturado temporalmente.

Soluciones:

```text
- Esperar unos segundos y reintentar
- Reducir RAG_TOP_K
- Reducir RAG_MAX_CONTEXT_CHARS
- Usar gemini-2.5-flash-lite
- Añadir fallback a otro proveedor como Groq
```

---

## 14. Git y seguridad

No subir al repositorio:

```text
.env
.venv/
__pycache__/
*.pyc
.DS_Store
__MACOSX/
backend/embeddings/
backend/chunks/
backend/documentos-parseados/
```

Opcionalmente, si no quieres subir PDFs al repo:

```text
backend/documentos/
```

Ejemplo de `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.DS_Store
__MACOSX/

backend/.env
backend/embeddings/
backend/chunks/
backend/documentos-parseados/

frontend/node_modules/
frontend/.env.local
```

Si alguna API key se subió a GitHub por error:

```text
1. Revocar la key
2. Crear una nueva
3. Actualizar .env local
```

---

## 15. Comandos útiles

### Activar entorno virtual desde backend

```bash
source ../.venv/bin/activate
```

### Comprobar que usas la venv correcta

```bash
which python
```

Debe salir algo parecido a:

```text
/Users/candeladavilamoreno/Documents/GitHub/Proyecto-Prueba-RAG/.venv/bin/python
```

### Comprobar configuración cargada

```bash
python -c "from config import GEMINI_MODEL, RAG_TOP_K, RAG_MAX_CONTEXT_CHARS; print(GEMINI_MODEL, RAG_TOP_K, RAG_MAX_CONTEXT_CHARS)"
```

Salida esperada:

```text
gemini-2.5-flash-lite 3 5000
```

### Arrancar backend

```bash
cd backend
source ../.venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Arrancar frontend

```bash
cd frontend
npm run dev
```

### Probar subida de PDF

```bash
curl -N -X POST "http://127.0.0.1:8000/upload-stream" \
  -F "file=@/ruta/a/tu/documento.pdf"
```

### Probar pregunta

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "cuando se fundó Mercedes-Benz", "top_k": 3}'
```

### Rehacer colección de Qdrant

```bash
cd backend
source ../.venv/bin/activate
python 05_subir_chunks_a_qdrant.py --recreate-collection
```

---

## 16. Estado actual del proyecto

El proyecto ya permite:

```text
- Crear frontend React con chat
- Subir PDFs desde la interfaz
- Mostrar progreso de procesamiento en el chat
- Guardar PDFs en backend/documentos/
- Extraer texto de PDFs
- Crear chunks con metadata
- Crear embeddings locales
- Crear/verificar colección de Qdrant
- Crear índice payload para document_id
- Borrar chunks antiguos por document_id
- Subir chunks nuevos a Qdrant
- Preguntar desde el chat
- Recuperar chunks relevantes
- Generar respuesta final con Gemini
- Mostrar fuentes de los chunks usados
```

---

## 17. Próximos pasos recomendados

```text
1. Mejorar diseño del chat
2. Añadir botón para borrar documentos
3. Añadir indicador de si Qdrant está conectado
4. Añadir fallback a Groq si Gemini devuelve 503
5. Convertir scripts en funciones Python en vez de ejecutarlos con subprocess
6. Desplegar frontend y backend
```
