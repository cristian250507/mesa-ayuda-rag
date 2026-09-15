"""Módulos 5, 7 y 8: Aumento, generación y post-procesamiento.

Orquesta el ciclo completo Retrieve -> Augment -> Generate, utilizando
Ollama de forma local para la reformulación y generación de respuestas.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

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


# --------------------------------------------------------------------------- #
# Configuración Ollama
# --------------------------------------------------------------------------- #

MODELO_CHAT = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Límite de contexto que se entrega al modelo.
# Esto evita enviar demasiado texto a Gemma.
MAX_CONTEXT_CHARS = 5000

# Máximo de caracteres permitidos para el historial.
MAX_HISTORY_CHARS = 2500


def crear_llm() -> ChatOllama:
    """Instancia el modelo de chat local de Ollama.

    Returns:
        Cliente ChatOllama configurado para utilizar Gemma 3:4b.
    """
    return ChatOllama(
        model=MODELO_CHAT,
        base_url=OLLAMA_BASE_URL,
        temperature=CONFIG_MODELO.temperature,
        num_predict=350,
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

        logger.info(
            "Agente inicializado con prompts %s y modelo Ollama %s.",
            VERSION_PROMPT,
            MODELO_CHAT,
        )

    # ------------------------------------------------------------------ #
    # Utilidades para reducir el prompt
    # ------------------------------------------------------------------ #

    @staticmethod
    def _limitar_texto(
        texto: str,
        max_caracteres: int,
    ) -> str:
        """Limita la cantidad de texto enviada al modelo."""
        if len(texto) <= max_caracteres:
            return texto

        return (
            texto[:max_caracteres]
            + "\n[CONTEXTO RECORTADO]"
        )

    # ------------------------------------------------------------------ #
    # Prompt chaining: reformulación
    # ------------------------------------------------------------------ #

    def _reformular(
        self,
        pregunta: str,
        historial: str,
    ) -> str:
        """Convierte una pregunta de seguimiento en una consulta autocontenida.

        Para evitar llamadas innecesarias al modelo, la reformulación solo se
        realiza cuando existe historial conversacional.
        """
        if historial.startswith("(inicio"):
            return pregunta

        historial_reducido = self._limitar_texto(
            historial,
            MAX_HISTORY_CHARS,
        )

        try:
            respuesta = self._llm.invoke(
                PROMPT_REFORMULACION.format(
                    historial=historial_reducido,
                    pregunta=pregunta,
                )
            )

            reformulada = str(
                respuesta.content
            ).strip()

            logger.info(
                "Consulta reformulada: '%s'",
                reformulada,
            )

            return reformulada or pregunta

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falló la reformulación, se usa la original: %s",
                exc,
            )

            return pregunta

    # ------------------------------------------------------------------ #
    # Post-procesamiento
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parsear_salida(
        bruto: str,
    ) -> Dict[str, Any]:
        """Valida y normaliza la salida JSON del modelo."""
        texto = (
            bruto.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            datos: Dict[str, Any] = json.loads(texto)

        except json.JSONDecodeError:
            logger.error(
                "El modelo no devolvió JSON válido. "
                "Se aplica salida segura."
            )

            return dict(RESPUESTA_SIN_CONTEXTO)

        campos = {
            "respuesta_encontrada",
            "respuesta",
            "fuentes",
            "nivel_confianza",
        }

        if not campos.issubset(datos.keys()):
            logger.error(
                "JSON incompleto: faltan campos obligatorios."
            )

            return dict(RESPUESTA_SIN_CONTEXTO)

        if (
            datos.get("respuesta_encontrada")
            and not datos.get("fuentes")
        ):
            logger.warning(
                "Respuesta afirmativa sin fuente. "
                "Se degrada a derivación."
            )

            return dict(RESPUESTA_SIN_CONTEXTO)

        datos.setdefault(
            "derivar_a_humano",
            False,
        )

        return datos

    # ------------------------------------------------------------------ #
    # Flujo principal
    # ------------------------------------------------------------------ #

    def responder(
        self,
        pregunta: str,
    ) -> Tuple[Dict[str, Any], List[Document]]:
        """Ejecuta el ciclo RAG completo para una consulta.

        Args:
            pregunta: Consulta en lenguaje natural del estudiante.

        Returns:
            Tupla (respuesta estructurada, chunks utilizados como evidencia).
        """
        if not pregunta.strip():
            return (
                dict(RESPUESTA_SIN_CONTEXTO),
                [],
            )

        # -------------------------------------------------------------- #
        # MEMORIA
        # -------------------------------------------------------------- #

        historial = self._memoria.obtener_historial()

        # -------------------------------------------------------------- #
        # QUERY REFORMULATION
        # -------------------------------------------------------------- #

        consulta = self._reformular(
            pregunta,
            historial,
        )

        # -------------------------------------------------------------- #
        # 1. RETRIEVE
        # -------------------------------------------------------------- #

        resultados = recuperar(
            self._indice,
            consulta,
        )

        if not resultados:
            respuesta = dict(
                RESPUESTA_SIN_CONTEXTO
            )

            self._memoria.registrar(
                pregunta,
                respuesta["respuesta"],
            )

            return respuesta, []

        # -------------------------------------------------------------- #
        # 2. AUGMENT
        # -------------------------------------------------------------- #

        contexto = formatear_contexto(
            resultados,
        )

        contexto = self._limitar_texto(
            contexto,
            MAX_CONTEXT_CHARS,
        )

        historial_reducido = self._limitar_texto(
            historial,
            MAX_HISTORY_CHARS,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            historial=historial_reducido,
            contexto=contexto,
            pregunta=consulta,
        )

        # Limitar también el prompt final como protección.
        user_prompt = self._limitar_texto(
            user_prompt,
            8500,
        )

        logger.info(
            "Prompt RAG preparado: %d caracteres.",
            len(user_prompt),
        )

        # -------------------------------------------------------------- #
        # 3. GENERATE
        # -------------------------------------------------------------- #

        try:
            salida = self._llm.invoke(
                [
                    {
                        "role": "system",
                        "content": construir_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error en la llamada al LLM local: %s",
                exc,
            )

            return (
                dict(RESPUESTA_SIN_CONTEXTO),
                [
                    doc
                    for doc, _ in resultados
                ],
            )

        texto_salida = str(
            salida.content
        ).strip()

        logger.info(
            "Gemma generó una respuesta de %d caracteres.",
            len(texto_salida),
        )

        respuesta = self._parsear_salida(
            texto_salida,
        )

        self._memoria.registrar(
            pregunta,
            respuesta["respuesta"],
        )

        return (
            respuesta,
            [
                doc
                for doc, _ in resultados
            ],
        )

    def reiniciar(self) -> None:
        """Reinicia la sesión conversacional."""
        self._memoria.reiniciar()