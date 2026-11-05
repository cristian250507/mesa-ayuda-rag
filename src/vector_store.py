"""Módulos 3 y 4: Embeddings, indexación vectorial y recuperación.

Se utiliza FAISS como base de datos vectorial local y Ollama para generar
embeddings sin depender de una API externa de pago.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config.settings import CONFIG_RAG, DIR_INDICE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuración Ollama
# --------------------------------------------------------------------------- #

MODELO_EMBEDDINGS = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


def obtener_embeddings() -> OllamaEmbeddings:
    """Construye el cliente de embeddings local de Ollama.

    Returns:
        Cliente de embeddings configurado con nomic-embed-text.
    """
    return OllamaEmbeddings(
        model=MODELO_EMBEDDINGS,
        base_url=OLLAMA_BASE_URL,
    )


def construir_indice(chunks: List[Document]) -> FAISS:
    """Genera los embeddings de los chunks y persiste el índice en disco.

    Args:
        chunks: Fragmentos producidos por el módulo de ingesta.

    Returns:
        Índice FAISS listo para consultas.

    Raises:
        ValueError: si la lista de chunks viene vacía.
        RuntimeError: si falla Ollama o la construcción del índice.
    """
    if not chunks:
        raise ValueError(
            "No se puede construir el índice: no hay chunks."
        )

    try:
        indice = FAISS.from_documents(
            chunks,
            obtener_embeddings(),
        )

        DIR_INDICE.mkdir(
            parents=True,
            exist_ok=True,
        )

        indice.save_local(
            str(DIR_INDICE),
        )

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error al construir el índice vectorial: {exc}"
        ) from exc

    logger.info(
        "Índice FAISS construido y guardado en '%s'.",
        DIR_INDICE,
    )

    return indice


def cargar_indice() -> FAISS:
    """Carga el índice FAISS previamente persistido.

    Returns:
        Índice FAISS.

    Raises:
        FileNotFoundError: si el índice aún no ha sido construido.
    """
    if not DIR_INDICE.exists():
        raise FileNotFoundError(
            "El índice no existe. Ejecuta primero "
            "'python -m src.main --indexar'."
        )

    return FAISS.load_local(
        str(DIR_INDICE),
        obtener_embeddings(),
        allow_dangerous_deserialization=True,
    )


def recuperar(
    indice: FAISS,
    consulta: str,
) -> List[Tuple[Document, float]]:
    """Recupera los chunks más similares aplicando umbral de similitud.

    Se calculan los puntajes ANTES de inyectar el contexto en el prompt. Si
    ningún fragmento supera el umbral, se devuelve lista vacía y el pipeline
    evita invocar al generador.

    Args:
        indice: Índice FAISS cargado.
        consulta: Pregunta autocontenida del usuario.

    Returns:
        Lista de tuplas (documento, distancia) que pasaron el filtro.
    """
    try:
        resultados = indice.similarity_search_with_score(
            consulta,
            k=CONFIG_RAG.top_k,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Fallo en la búsqueda vectorial: %s",
            exc,
        )
        return []

    filtrados = [
        (doc, float(score))
        for doc, score in resultados
        if float(score) <= CONFIG_RAG.umbral_distancia
    ]

    logger.info(
        "Recuperación: %d candidatos, %d superaron el umbral (%.2f).",
        len(resultados),
        len(filtrados),
        CONFIG_RAG.umbral_distancia,
    )

    return filtrados


def formatear_contexto(
    resultados: List[Tuple[Document, float]],
) -> str:
    """Ensambla el bloque CONTEXTO con metadatos explícitos de trazabilidad.

    Args:
        resultados: Salida de :func:`recuperar`.

    Returns:
        Cadena delimitada lista para inyectar en el user prompt.
    """
    bloques: List[str] = []

    for doc, score in resultados:
        meta = doc.metadata

        encabezado = (
            f"[FUENTE: {meta.get('documento', 'desconocido')} "
            f"| TIPO: {meta.get('tipo', 'interno')} "
            f"| PÁGINA: {meta.get('pagina', 's/n')} "
            f"| SIMILITUD: {1 / (1 + score):.2f}]"
        )

        bloques.append(
            f"{encabezado}\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(bloques)