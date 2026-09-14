"""Módulo 1 y 2: Ingesta de documentos y fragmentación (chunking).

Responsabilidades:
    * Cargar documentos desde fuentes INTERNAS y EXTERNAS.
    * Adjuntar el documento de origen como metadato desde la ingesta
      (regla arquitectónica de trazabilidad de datos en RAG).
    * Fragmentar el texto con RecursiveCharacterTextSplitter propagando
      los metadatos a cada chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config.settings import CONFIG_RAG, DIR_EXTERNO, DIR_INTERNO

logger = logging.getLogger(__name__)

EXTENSIONES_SOPORTADAS: tuple[str, ...] = (".pdf", ".txt", ".md")


def _cargar_archivo(ruta: Path, tipo_fuente: str) -> List[Document]:
    """Carga un archivo individual y le adjunta metadatos de trazabilidad.

    Args:
        ruta: Ruta absoluta del archivo a cargar.
        tipo_fuente: 'interno' o 'externo'.

    Returns:
        Lista de documentos LangChain con metadatos enriquecidos. Lista vacía
        si el archivo no pudo procesarse.
    """
    try:
        if ruta.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(ruta))
        else:
            loader = TextLoader(str(ruta), encoding="utf-8")

        documentos: List[Document] = loader.load()

    except FileNotFoundError:
        logger.error("Archivo no encontrado: %s", ruta)
        return []
    except Exception as exc:  # noqa: BLE001 - la ingesta no debe tumbar el pipeline
        logger.error("Error al cargar '%s': %s", ruta.name, exc)
        return []

    for doc in documentos:
        doc.metadata.update(
            {
                "documento": ruta.stem,
                "tipo": tipo_fuente,
                "archivo": ruta.name,
                "pagina": doc.metadata.get("page", 0) + 1,
            }
        )

    logger.info("Cargado '%s' (%s) -> %d página(s).", ruta.name, tipo_fuente, len(documentos))
    return documentos


def cargar_corpus() -> List[Document]:
    """Carga todos los documentos internos y externos disponibles.

    Returns:
        Lista consolidada de documentos con metadatos de origen.

    Raises:
        FileNotFoundError: si no se encontró ningún documento válido.
    """
    corpus: List[Document] = []

    for directorio, tipo in ((DIR_INTERNO, "interno"), (DIR_EXTERNO, "externo")):
        if not directorio.exists():
            logger.warning("El directorio '%s' no existe. Se omite.", directorio)
            continue

        for ruta in sorted(directorio.iterdir()):
            if ruta.suffix.lower() in EXTENSIONES_SOPORTADAS:
                corpus.extend(_cargar_archivo(ruta, tipo))

    if not corpus:
        raise FileNotFoundError(
            "No se encontraron documentos en data/interno ni en data/externo."
        )

    logger.info("Corpus total: %d documento(s) cargado(s).", len(corpus))
    return corpus


def fragmentar(documentos: List[Document]) -> List[Document]:
    """Divide los documentos en chunks preservando sentido semántico.

    El separador recursivo intenta cortar primero por párrafo, luego por línea
    y finalmente por carácter, de modo que un artículo reglamentario rara vez
    queda partido. El solapamiento cubre los casos en que sí ocurre.

    Args:
        documentos: Documentos cargados por :func:`cargar_corpus`.

    Returns:
        Lista de fragmentos con los metadatos del documento padre.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG_RAG.chunk_size,
        chunk_overlap=CONFIG_RAG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Document] = splitter.split_documents(documentos)

    for indice, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = indice

    logger.info(
        "Fragmentación completada: %d chunks (size=%d, overlap=%d).",
        len(chunks),
        CONFIG_RAG.chunk_size,
        CONFIG_RAG.chunk_overlap,
    )
    return chunks
