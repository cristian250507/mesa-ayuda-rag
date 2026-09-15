"""Gestión de memoria conversacional del agente RAG.

Mantiene los últimos turnos de la conversación y genera un resumen
cuando se alcanza el número máximo de turnos configurado.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from config.settings import CONFIG_MEMORIA

logger = logging.getLogger(__name__)


class GestorMemoria:
    """Gestiona el historial conversacional del agente."""

    def __init__(self, llm: Any) -> None:
        """Inicializa el gestor de memoria.

        Args:
            llm: Modelo de lenguaje utilizado para generar resúmenes.
        """
        self._llm = llm

        self._historial: List[Tuple[str, str]] = []

        self._resumen: str = ""

        logger.info(
            "Gestor de memoria inicializado."
        )

    def registrar(
        self,
        pregunta: str,
        respuesta: str,
    ) -> None:
        """Registra un intercambio de la conversación.

        Args:
            pregunta: Pregunta realizada por el usuario.
            respuesta: Respuesta entregada por el agente.
        """
        self._historial.append(
            (
                pregunta.strip(),
                respuesta.strip(),
            )
        )

        # Mantener solamente la ventana configurada.
        max_turnos = CONFIG_MEMORIA.ventana_turnos

        if len(self._historial) > max_turnos:
            self._historial = self._historial[-max_turnos:]

        logger.info(
            "Turno registrado. Historial actual: %d turnos.",
            len(self._historial),
        )

        # Generar resumen cuando se alcanza el umbral.
        if len(self._historial) >= CONFIG_MEMORIA.umbral_resumen:
            self._generar_resumen()

    def _generar_resumen(self) -> None:
        """Genera un resumen de la conversación utilizando el LLM."""
        if not self._historial:
            return

        conversaciones = "\n".join(
            f"Usuario: {pregunta}\nAsistente: {respuesta}"
            for pregunta, respuesta in self._historial
        )

        prompt = f"""
Resume brevemente la siguiente conversación académica.

Conserva solamente información útil para responder futuras preguntas:
- temas consultados;
- datos importantes;
- contexto que pueda necesitarse posteriormente.

No inventes información.

CONVERSACIÓN:
{conversaciones}

RESUMEN:
""".strip()

        try:
            resultado = self._llm.invoke(prompt)

            self._resumen = str(
                resultado.content
            ).strip()

            logger.info(
                "Resumen conversacional actualizado."
            )

        except Exception as exc:
            logger.warning(
                "No se pudo generar el resumen: %s",
                exc,
            )

    def obtener_historial(self) -> str:
        """Devuelve el contexto disponible de la conversación.

        Returns:
            Texto con resumen e historial reciente.
        """
        partes: List[str] = []

        if self._resumen:
            partes.append(
                f"RESUMEN DE LA CONVERSACIÓN:\n{self._resumen}"
            )

        if self._historial:
            conversaciones = "\n".join(
                f"Usuario: {pregunta}\n"
                f"Asistente: {respuesta}"
                for pregunta, respuesta in self._historial
            )

            partes.append(
                f"HISTORIAL RECIENTE:\n{conversaciones}"
            )

        if not partes:
            return "(inicio de la conversación)"

        return "\n\n".join(partes)

    def reiniciar(self) -> None:
        """Reinicia completamente la memoria."""
        self._historial.clear()
        self._resumen = ""

        logger.info(
            "Memoria conversacional reiniciada."
        )