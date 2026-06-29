# Proyecto Prueba RAG con LangChain, Embeddings y Qdrant

Este proyecto implementa un pipeline básico de RAG usando documentos PDF como fuente de información.

El flujo principal es:

```text
PDF
↓gi 
Extracción de texto
↓
Extracción página a página
↓
Chunking con LangChain
↓
Embeddings locales con sentence-transformers
↓
Subida a Qdrant
↓
Búsqueda semántica
```

Actualmente el proyecto cubre la parte de **ingesta, chunking, embeddings y búsqueda en Qdrant**.
El siguiente paso será añadir la generación de respuesta final con un LLM.

---

## Estructura del proyecto

```text
Proyecto-Prueba-RAG/
├── documentos/
│   └── archivos PDF originales
│
├── documentos-parseados/
│   ├── documento.txt
│   └── documento.pages.jsonl
│
├── chunks/
│   └── chunks.jsonl
│
├── embeddings/
│   └── embeddings.npy
│
├── 01_liteparse_pdf_a_txt.py
├── 02_langchain_chunking.py
├── 03_embeddings_locales.py
├── 04_busqueda_semantica.py
├── 05_subir_chunks_a_qdrant.py
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 1. Crear entorno virtual

```bash
python -m venv .venv
```

Activar entorno en macOS/Linux:

```bash
source .venv/bin/activate
```

Activar entorno en Windows:

```bash
.venv\Scripts\activate
```

---

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto.

Puedes usar `.env.example` como referencia:

```env
QDRANT_URL=https://TU_CLUSTER.qdrant.io
QDRANT_API_KEY=TU_API_KEY
QDRANT_COLLECTION=documentos_rag
```

Importante: el archivo `.env` no debe subirse a GitHub.

---

## 4. Añadir documentos

Coloca tus PDFs dentro de la carpeta:

```text
documentos/
```

Ejemplo:

```text
documentos/
└── mercedes_historia_extensa.pdf
```

---

## 5. Extraer texto de los PDFs

Ejecuta:

```bash
python 01_liteparse_pdf_a_txt.py
```

Este script genera dos tipos de salida en `documentos-parseados/`:

```text
documentos-parseados/documento.txt
documentos-parseados/documento.pages.jsonl
```

El archivo `.txt` contiene el texto completo del documento.

El archivo `.pages.jsonl` contiene la extracción página a página, con metadata como:

```json
{
  "document_id": "mercedes_historia_extensa",
  "filename": "mercedes_historia_extensa.pdf",
  "source": "mercedes_historia_extensa.pdf",
  "source_type": "pdf",
  "page": 1,
  "page_label": "1",
  "text": "Texto extraído de la página..."
}
```

Este archivo es el que se usa para crear chunks con referencias correctas de página.

---

## 6. Crear chunks con LangChain

Ejecuta:

```bash
python 02_langchain_chunking.py
```

Este script lee los archivos:

```text
documentos-parseados/*.pages.jsonl
```

y genera:

```text
chunks/chunks.jsonl
```

Cada chunk incluye el texto y metadata útil para RAG:

```json
{
  "id": "mercedes_historia_extensa_p001_c000",
  "text": "Texto del chunk...",
  "metadata": {
    "source": "mercedes_historia_extensa.pdf",
    "filename": "mercedes_historia_extensa.pdf",
    "document_id": "mercedes_historia_extensa",
    "source_type": "pdf",
    "page": 1,
    "page_label": "1",
    "language": "es",
    "chunk_index": 0,
    "chunk_index_in_page": 0,
    "chunk_size": 923
  }
}
```

La metadata permite saber de qué documento y página viene cada fragmento.

---

## 7. Generar embeddings locales

Ejecuta:

```bash
python 03_embeddings_locales.py
```

Este script lee:

```text
chunks/chunks.jsonl
```

y genera embeddings locales en:

```text
embeddings/embeddings.npy
```

El modelo usado está definido en `config.py`.

Por defecto se recomienda usar un modelo multilingüe:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Este modelo genera vectores de 384 dimensiones.

---

## 8. Probar búsqueda semántica local

Antes de subir a Qdrant, puedes probar la búsqueda local:

```bash
python 04_busqueda_semantica.py
```

Este script usa los embeddings generados localmente y devuelve los chunks más parecidos a una pregunta.

---

## 9. Subir chunks a Qdrant

Ejecuta:

```bash
python 05_subir_chunks_a_qdrant.py
```

Este script hace lo siguiente:

```text
1. Lee chunks/chunks.jsonl
2. Crea embeddings para los chunks
3. Comprueba si existe la colección en Qdrant
4. Valida que la dimensión vectorial sea correcta
5. Borra de Qdrant los chunks antiguos de los documentos que se están reindexando
6. Sube los chunks nuevos
```

Por defecto, el script **no borra toda la colección**.

Solo borra los chunks antiguos de los documentos presentes en `chunks/chunks.jsonl`, usando el campo:

```text
document_id
```

Esto permite reindexar un documento sin borrar otros documentos de la colección.

---

## 10. Probar búsqueda en Qdrant

Puedes subir los chunks y lanzar una query de prueba con:

```bash
python 05_subir_chunks_a_qdrant.py --query "quien fue Bertha Benz" --top-k 5
```

También puedes probar otras preguntas:

```bash
python 05_subir_chunks_a_qdrant.py --query "qué relación tiene Mercedes-Benz con la innovación" --top-k 5
```

---

## 11. Rehacer completamente la colección de Qdrant

Si quieres borrar toda la colección y crearla de nuevo desde cero, ejecuta:

```bash
python 05_subir_chunks_a_qdrant.py --recreate-collection
```

También puedes recrearla y probar una query en el mismo comando:

```bash
python 05_subir_chunks_a_qdrant.py --recreate-collection --query "quien fue Bertha Benz" --top-k 5
```

Este comando es útil cuando:

```text
- Cambias el modelo de embeddings
- Cambias la dimensión de los vectores
- Quieres limpiar datos de prueba
- La colección quedó inconsistente
- Quieres empezar la demo desde cero
```

Importante: `--recreate-collection` borra toda la colección configurada en `QDRANT_COLLECTION`.

---

## 12. Orden recomendado de ejecución

Para ejecutar todo el pipeline desde cero:

```bash
python 01_liteparse_pdf_a_txt.py
python 02_langchain_chunking.py
python 03_embeddings_locales.py
python 04_busqueda_semantica.py
python 05_subir_chunks_a_qdrant.py
```

Para rehacer todo y recrear Qdrant desde cero:

```bash
python 01_liteparse_pdf_a_txt.py
python 02_langchain_chunking.py
python 05_subir_chunks_a_qdrant.py --recreate-collection
```

Para rehacer todo y probar una búsqueda en Qdrant:

```bash
python 01_liteparse_pdf_a_txt.py
python 02_langchain_chunking.py
python 05_subir_chunks_a_qdrant.py --recreate-collection --query "quien fue Bertha Benz" --top-k 5
```

---

## Limpieza de Qdrant

Hay dos formas de limpiar datos en Qdrant.

### Opción segura: limpiar solo documentos reindexados

```bash
python 05_subir_chunks_a_qdrant.py
```

Este comando borra solo los chunks antiguos cuyo `document_id` coincide con los documentos actuales de `chunks/chunks.jsonl`.

Es la opción recomendada para el uso normal.

### Opción agresiva: borrar toda la colección

```bash
python 05_subir_chunks_a_qdrant.py --recreate-collection
```

Este comando borra la colección completa y la crea de nuevo.

Úsalo si quieres empezar desde cero.

---

## Metadata usada en los chunks

Cada chunk incluye metadata para poder mostrar fuentes en las respuestas del RAG.

Campos principales:

```text
source              Nombre del documento fuente
filename            Nombre del archivo original
document_id         Identificador estable del documento
source_type         Tipo de fuente, por ejemplo pdf
page                Número de página
page_label          Etiqueta de página
language            Idioma del documento
chunk_index         Índice global del chunk
chunk_index_in_page Índice del chunk dentro de la página
chunk_size          Tamaño del chunk en caracteres
```

Ejemplo:

```json
{
  "source": "mercedes_historia_extensa.pdf",
  "filename": "mercedes_historia_extensa.pdf",
  "document_id": "mercedes_historia_extensa",
  "source_type": "pdf",
  "page": 3,
  "page_label": "3",
  "language": "es",
  "chunk_index": 14,
  "chunk_index_in_page": 2,
  "chunk_size": 845
}
```

---

## Seguridad

No subas estos archivos o carpetas al repositorio:

```text
.env
.venv/
__pycache__/
embeddings/
chunks/
documentos-parseados/
__MACOSX/
.DS_Store
```

El archivo `.env` contiene credenciales privadas como la API key de Qdrant.

---

## Estado actual del proyecto

El proyecto ya permite:

```text
- Leer PDFs
- Extraer texto completo
- Extraer texto página a página
- Crear chunks con metadata
- Generar embeddings
- Subir chunks a Qdrant
- Limpiar chunks antiguos por document_id
- Recrear la colección de Qdrant
- Validar dimensión vectorial de la colección
- Buscar chunks relevantes en Qdrant
```

---

## Siguiente paso

El siguiente paso es crear:

```text
06_ask_rag.py
```

Ese script hará:

```text
Pregunta del usuario
↓
Embedding de la pregunta
↓
Búsqueda en Qdrant
↓
Construcción del contexto
↓
Llamada al LLM
↓
Respuesta final con fuentes
```
