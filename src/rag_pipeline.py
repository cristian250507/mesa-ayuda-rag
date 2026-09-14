"""Módulos 5, 7 y 8: Aumento, generación y post-procesamiento.

Orquesta el ciclo completo Retrieve -> Augment -> Generate, incorporando
prompt chaining para la reformulación de la consulta y validación estructural
de la salida antes de devolverla al usuario.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config.settings import CONFIG_MODELO
from src.memoria import GestorMemoria
from src.prompts import (
    PROMPT_REFORMULACION,
    RESPUESTA_SIN_CONTEXTO,
    USER_PROMPT_TEMPLATE,
    VERSION_PROMPT,
    construir_system_prompt,
)
from src.vector_store import formatear_contexto, recuperar

logger = logging.getLogger(__name__)


def crear_llm() -> ChatOpenAI:
    """Instancia el cliente de chat contra GitHub Models.

    Returns:
        Cliente configurado con temperatura baja para respuestas deterministas.
    """
    return ChatOpenAI(
        model=CONFIG_MODELO.modelo_chat,
        base_url=CONFIG_MODELO.base_url,
        api_key=CONFIG_MODELO.api_key,
        temperature=CONFIG_MODELO.temperature,
        max_tokens=CONFIG_MODELO.max_tokens,
        timeout=CONFIG_MODELO.timeout,
    )


class AgenteNormativo:
    """Agente conversacional RAG para consultas normativas académicas."""

    def __init__(self, indice: Any) -> None:
        """Inicializa el agente.

        Args:
            indice: Índice FAISS ya cargado o construido.
        """
        self._indice = indice
        self._llm = crear_llm()
        self._memoria = GestorMemoria(self._llm)
        logger.info("Agente inicializado con prompts %s.", VERSION_PROMPT)

    # ------------------------------------------------------------------ #
    # Prompt chaining: reformulación previa a la búsqueda vectorial
    # ------------------------------------------------------------------ #
    def _reformular(self, pregunta: str, historial: str) -> str:
        """Convierte una pregunta de seguimiento en una consulta autocontenida.

        Sin este paso, una pregunta como "¿y si es la segunda vez?" genera un
        embedding sin carga semántica y la recuperación falla.

        Args:
            pregunta: Pregunta original del estudiante.
            historial: Historial conversacional vigente.

        Returns:
            Pregunta autocontenida, o la original si la reformulación falla.
        """
        if historial.startswith("(inicio"):
            return pregunta

        try:
            respuesta = self._llm.invoke(
                PROMPT_REFORMULACION.format(historial=historial, pregunta=pregunta)
            )
            reformulada = str(respuesta.content).strip()
            logger.info("Consulta reformulada: '%s'", reformulada)
            return reformulada or pregunta
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallo la reformulación, se usa la original: %s", exc)
            return pregunta

    # ------------------------------------------------------------------ #
    # Post-procesamiento
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parsear_salida(bruto: str) -> Dict[str, Any]:
        """Valida y normaliza la salida JSON del modelo.

        Args:
            bruto: Texto devuelto por el LLM.

        Returns:
            Diccionario con la estructura esperada. Ante cualquier anomalía se
            devuelve la respuesta segura de derivación.
        """
        texto = bruto.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

        try:
            datos: Dict[str, Any] = json.loads(texto)
        except json.JSONDecodeError:
            logger.error("El modelo no devolvió JSON válido. Se aplica salida segura.")
            return dict(RESPUESTA_SIN_CONTEXTO)

        campos = {"respuesta_encontrada", "respuesta", "fuentes", "nivel_confianza"}
        if not campos.issubset(datos.keys()):
            logger.error("JSON incompleto: faltan campos obligatorios.")
            return dict(RESPUESTA_SIN_CONTEXTO)

        # Regla de trazabilidad: no se admite respuesta afirmativa sin fuente.
        if datos.get("respuesta_encontrada") and not datos.get("fuentes"):
            logger.warning("Respuesta afirmativa sin fuentes. Se degrada a derivación.")
            return dict(RESPUESTA_SIN_CONTEXTO)

        datos.setdefault("derivar_a_humano", False)
        return datos

    # ------------------------------------------------------------------ #
    # Flujo principal
    # ------------------------------------------------------------------ #
    def responder(self, pregunta: str) -> Tuple[Dict[str, Any], List[Document]]:
        """Ejecuta el ciclo RAG completo para una consulta.

        Args:
            pregunta: Consulta en lenguaje natural del estudiante.

        Returns:
            Tupla (respuesta estructurada, chunks utilizados como evidencia).
        """
        if not pregunta.strip():
            return dict(RESPUESTA_SIN_CONTEXTO), []

        historial = self._memoria.obtener_historial()
        consulta = self._reformular(pregunta, historial)

        # 1. RETRIEVE (con corte por umbral de similitud)
        resultados = recuperar(self._indice, consulta)
        if not resultados:
            respuesta = dict(RESPUESTA_SIN_CONTEXTO)
            self._memoria.registrar(pregunta, respuesta["respuesta"])
            return respuesta, []

        # 2. AUGMENT
        contexto = formatear_contexto(resultados)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            historial=historial, contexto=contexto, pregunta=consulta
        )

        # 3. GENERATE
        try:
            salida = self._llm.invoke(
                [
                    {"role": "system", "content": construir_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en la llamada al LLM: %s", exc)
            return dict(RESPUESTA_SIN_CONTEXTO), [doc for doc, _ in resultados]

        respuesta = self._parsear_salida(str(salida.content))
        self._memoria.registrar(pregunta, respuesta["respuesta"])

        return respuesta, [doc for doc, _ in resultados]

    def reiniciar(self) -> None:
        """Reinicia la sesión conversacional."""
        self._memoria.reiniciar()
