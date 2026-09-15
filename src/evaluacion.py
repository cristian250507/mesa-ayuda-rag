"""Módulo 9: Evaluación del sistema RAG.

Implementa las cuatro métricas vistas en la asignatura:
    Recuperación -> Context Precision, Context Recall
    Generación   -> Faithfulness, Answer Relevancy

Las dos primeras se calculan de forma determinista sobre el conjunto dorado.
Las dos últimas se estiman mediante LLM-as-a-judge, que es el enfoque que usan
herramientas como RAGAS y LangSmith.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

PROMPT_JUEZ = """\
Actúa como evaluador imparcial de sistemas RAG. No des explicaciones.

CONTEXTO ENTREGADO AL MODELO:
{contexto}

PREGUNTA: {pregunta}
RESPUESTA GENERADA: {respuesta}

Evalúa paso a paso de forma interna y devuelve estrictamente este JSON:
{{"faithfulness": <0.0-1.0>, "answer_relevancy": <0.0-1.0>}}

faithfulness = proporción de afirmaciones de la respuesta verificables
literalmente en el contexto.
answer_relevancy = qué tan directa y útil es la respuesta frente a la pregunta.
"""


def context_precision(recuperados: List[Document], relevantes: List[str]) -> float:
    """Proporción de documentos recuperados que son efectivamente relevantes.

    Args:
        recuperados: Chunks devueltos por el retriever.
        relevantes: Nombres de documentos considerados relevantes (gold set).

    Returns:
        Valor entre 0.0 y 1.0. Una precisión baja implica ruido en el prompt.
    """
    if not recuperados:
        return 0.0

    aciertos = sum(
        1 for doc in recuperados if doc.metadata.get("documento") in relevantes
    )
    return aciertos / len(recuperados)


def context_recall(recuperados: List[Document], relevantes: List[str]) -> float:
    """Proporción de documentos relevantes existentes que fueron recuperados.

    Args:
        recuperados: Chunks devueltos por el retriever.
        relevantes: Nombres de documentos considerados relevantes (gold set).

    Returns:
        Valor entre 0.0 y 1.0. Un recall bajo implica información faltante.
    """
    if not relevantes:
        return 1.0

    encontrados = {
        doc.metadata.get("documento")
        for doc in recuperados
        if doc.metadata.get("documento") in relevantes
    }
    return len(encontrados) / len(set(relevantes))


def evaluar_generacion(
    llm: ChatOpenAI, pregunta: str, respuesta: str, contexto: str
) -> Dict[str, float]:
    """Estima Faithfulness y Answer Relevancy mediante LLM-as-a-judge.

    Args:
        llm: Cliente de chat usado como evaluador.
        pregunta: Consulta original.
        respuesta: Respuesta generada por el agente.
        contexto: Contexto que se inyectó al generar.

    Returns:
        Diccionario con ambas métricas; ceros si la evaluación falla.
    """
    try:
        salida = llm.invoke(
            PROMPT_JUEZ.format(contexto=contexto, pregunta=pregunta, respuesta=respuesta)
        )
        texto = str(salida.content).strip().removeprefix("```json").removesuffix("```")
        datos = json.loads(texto)
        return {
            "faithfulness": float(datos.get("faithfulness", 0.0)),
            "answer_relevancy": float(datos.get("answer_relevancy", 0.0)),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.error("Respuesta del juez no parseable: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error al evaluar la generación: %s", exc)

    return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def cargar_conjunto_dorado(ruta: Path) -> List[Dict[str, Any]]:
    """Carga el conjunto de preguntas de evaluación desde un JSON.

    Args:
        ruta: Ruta al archivo del gold set.

    Returns:
        Lista de casos de prueba. Lista vacía si el archivo no existe.
    """
    try:
        with ruta.open(encoding="utf-8") as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        logger.error("No se encontró el conjunto dorado en '%s'.", ruta)
        return []
    except json.JSONDecodeError as exc:
        logger.error("El conjunto dorado no es JSON válido: %s", exc)
        return []


def resumir(resultados: List[Dict[str, float]]) -> Dict[str, float]:
    """Promedia las métricas de todos los casos evaluados.

    Args:
        resultados: Métricas por caso.

    Returns:
        Promedio de cada métrica.
    """
    if not resultados:
        return {}

    claves = resultados[0].keys()
    return {
        clave: round(sum(r[clave] for r in resultados) / len(resultados), 3)
        for clave in claves
    }
