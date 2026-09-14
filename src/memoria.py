"""Módulo 6: Gestión del contexto conversacional.

Estrategia híbrida:
    * ConversationBufferWindowMemory(k=5) como memoria operativa, para mantener
      el hilo reciente sin inflar el consumo de tokens en cada turno.
    * ConversationSummaryMemory a partir de cierto número de turnos, para
      sesiones largas de atención sin perder el contexto inicial.
"""

from __future__ import annotations

import logging
from typing import List

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from config.settings import CONFIG_MEMORIA

logger = logging.getLogger(__name__)


class GestorMemoria:
    """Administra el historial conversacional con conmutación automática."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Inicializa el gestor.

        Args:
            llm: Cliente de chat utilizado por la memoria de resumen.
        """
        self._llm = llm
        self._turnos: int = 0

        self._ventana = ConversationBufferWindowMemory(
            k=CONFIG_MEMORIA.ventana_turnos,
            return_messages=False,
            memory_key="historial",
        )
        self._resumen = ConversationSummaryMemory(
            llm=llm,
            return_messages=False,
            memory_key="historial",
        )

    @property
    def modo(self) -> str:
        """Devuelve el modo de memoria activo ('ventana' o 'resumen')."""
        return "resumen" if self._turnos >= CONFIG_MEMORIA.umbral_resumen else "ventana"

    def registrar(self, pregunta: str, respuesta: str) -> None:
        """Guarda un turno completo en ambas memorias.

        Args:
            pregunta: Consulta del estudiante.
            respuesta: Respuesta entregada por el agente.
        """
        entrada = {"input": pregunta}
        salida = {"output": respuesta}

        try:
            self._ventana.save_context(entrada, salida)
            if self.modo == "resumen" or self._turnos >= CONFIG_MEMORIA.umbral_resumen - 1:
                self._resumen.save_context(entrada, salida)
        except Exception as exc:  # noqa: BLE001 - la memoria nunca debe cortar la atención
            logger.warning("No se pudo guardar el turno en memoria: %s", exc)

        self._turnos += 1

    def obtener_historial(self) -> str:
        """Devuelve el historial en el formato adecuado al modo activo.

        Returns:
            Texto del historial, o un marcador si la sesión recién comienza.
        """
        try:
            memoria = self._resumen if self.modo == "resumen" else self._ventana
            historial = memoria.load_memory_variables({}).get("historial", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo recuperar el historial: %s", exc)
            return "(sin historial disponible)"

        return historial.strip() or "(inicio de la conversación)"

    def reiniciar(self) -> None:
        """Limpia el historial para comenzar una nueva atención."""
        self._ventana.clear()
        self._resumen.clear()
        self._turnos = 0
        logger.info("Memoria conversacional reiniciada.")

    def turnos(self) -> int:
        """Número de turnos registrados en la sesión actual."""
        return self._turnos


__all__: List[str] = ["GestorMemoria"]
