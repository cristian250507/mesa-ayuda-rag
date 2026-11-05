"""Módulo de Herramientas (Tools) del Agente LangGraph.

Define las herramientas de:
1. Consulta (RAG sobre FAISS).
2. Escritura (Creación de Tickets en JSON).
3. Razonamiento (Cálculo de causales de eliminación).

Asignatura: ISY0101 - Ingeniería de Soluciones con IA (EP2)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from config.settings import DIR_DATA
from src.vector_store import cargar_indice, formatear_contexto, recuperar

logger = logging.getLogger(__name__)

# Intentar cargar el índice FAISS para la herramienta de consulta
_INDICE_FAISS = None


def _obtener_indice():
    global _INDICE_FAISS
    if _INDICE_FAISS is None:
        try:
            _INDICE_FAISS = cargar_indice()
        except Exception as exc:
            logger.warning("No se pudo cargar el índice FAISS en tools: %s", exc)
            _INDICE_FAISS = None
    return _INDICE_FAISS


@tool
def consultar_normativa_academica(consulta: str) -> str:
    """Consulta el reglamento académico, calendario y normativa de beneficios.

    Usa esta herramienta cuando el estudiante pregunte por reglas, plazos,
    requisitos, causales de eliminación o condiciones de becas.
    """
    indice = _obtener_indice()
    if indice is None:
        return "Error: El índice vectorial no ha sido construido. Ejecuta primero '--indexar'."

    resultados = recuperar(indice, consulta)
    if not resultados:
        return "No se encontró información relevante sobre esa consulta en la documentación oficial."

    return formatear_contexto(resultados)


@tool
def crear_ticket_atencion(rut_estudiante: str, motivo: str, detalle: str) -> str:
    """Crea un ticket de atención presencial o derivación en la DAE.

    Usa esta herramienta cuando la consulta requiera atención presencial, cuando
    exista una contradicción o cuando el estudiante solicite apelar o ser derivado a un ejecutivo.
    """
    ruta_tickets = DIR_DATA / "tickets.json"

    ticket = {
        "id_ticket": f"TCK-{int(datetime.now().timestamp())}",
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rut_estudiante": rut_estudiante or "NO_ESPECIFICADO",
        "motivo": motivo,
        "detalle": detalle,
        "estado": "PENDIENTE_EJECUTIVO",
    }

    tickets = []
    if ruta_tickets.exists():
        with open(ruta_tickets, "r", encoding="utf-8") as f:
            try:
                tickets = json.load(f)
            except json.JSONDecodeError:
                tickets = []

    tickets.append(ticket)
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    with open(ruta_tickets, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)

    return f"Ticket creado con éxito. Folio: {ticket['id_ticket']}. Se ha derivado tu caso a la Dirección de Asuntos Estudiantiles."


@tool
def evaluar_causal_eliminacion(
    ramos_inscritos: int,
    ramos_reprobados: int,
    es_segunda_reprobacion_mismo_ramo: bool,
) -> str:
    """Calcula si la situación académica de un estudiante cae en causal de eliminación.

    Usa esta herramienta cuando el estudiante entregue datos sobre cantidad de ramos aprobados/reprobados.
    """
    if es_segunda_reprobacion_mismo_ramo:
        return (
            "SITUACIÓN CRÍTICA: Incurre en Causal de Eliminación (Art. 24 del Reglamento Académico). "
            "Motivo: Reprobar por segunda vez consecutiva una misma asignatura."
        )

    porcentaje_reprobacion = (
        (ramos_reprobados / ramos_inscritos) * 100 if ramos_inscritos > 0 else 0
    )

    if porcentaje_reprobacion > 50:
        return (
            f"SITUACIÓN CRÍTICA: Alerta de Causal de Eliminación (Art. 25 del Reglamento Académico). "
            f"Porcentaje de reprobación en el semestre: {porcentaje_reprobacion:.1f}%. "
            f"Si esto se repite durante dos semestres consecutivos, incurre en causal de eliminación formal."
        )

    return (
        f"SITUACIÓN REGULAR: Porcentaje de reprobación del semestre: {porcentaje_reprobacion:.1f}% "
        f"(dentro del margen permitido de hasta 50%)."
    )