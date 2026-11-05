
"""Módulo Orquestador del Agente LangGraph.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA (EP2)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config.settings import CONFIG_MODELO
from src.tools import (
    consultar_normativa_academica,
    crear_ticket_atencion,
    evaluar_causal_eliminacion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PROMPT DEL SISTEMA
# ---------------------------------------------------------------------------

PROMPT_SISTEMA_AGENTE = """\
Eres el Agente Virtual Oficial de la Mesa de Ayuda Académica de la institución.

REGLAS DE OPERACIÓN Y USO DE HERRAMIENTAS:

1. CONSULTAS SOBRE LA CONVERSACIÓN O HISTORIAL:
   - Si el estudiante pregunta por un folio, ticket, solicitud o información
     entregada anteriormente, responde utilizando el historial de conversación.
   - No inventes folios ni información anterior.
   - No invoques herramientas para recuperar información del historial.

2. USO DE HERRAMIENTAS PARA NUEVAS CONSULTAS:
   - Normativa y plazos de asignaturas: invoca consultar_normativa_academica.
   - Evaluación de causal de eliminación: invoca evaluar_causal_eliminacion.
   - Crear un ticket formal: invoca crear_ticket_atencion.

3. RESTRICCIONES:
   - Nunca repitas etiquetas técnicas como [RUT...] en tu respuesta.
   - Nunca muestres JSON de llamadas a herramientas al estudiante.
   - No inventes información normativa ni resultados de herramientas.
   - Responde en español, de forma clara y amable.
"""


class AgenteOrquestador:
    """Clase principal que envuelve el agente ReAct de LangGraph."""

    def __init__(self) -> None:

        # Inicialización del modelo Ollama.
        self.llm = ChatOllama(
            model=CONFIG_MODELO.modelo_chat,
            base_url=CONFIG_MODELO.ollama_base_url,
            temperature=CONFIG_MODELO.temperature,
        )

        # Herramientas disponibles.
        self.herramientas = [
            consultar_normativa_academica,
            evaluar_causal_eliminacion,
            crear_ticket_atencion,
        ]

        # Memoria de corto plazo de LangGraph.
        self.checkpointer = MemorySaver()

        # Inicialización del agente.
        self.agente = create_react_agent(
            model=self.llm,
            tools=self.herramientas,
            checkpointer=self.checkpointer,
            prompt=PROMPT_SISTEMA_AGENTE,
        )

        logger.info(
            "Agente LangGraph ReAct inicializado con %d herramientas y memoria.",
            len(self.herramientas),
        )

    # -----------------------------------------------------------------------
    # RECUPERACIÓN DE INFORMACIÓN DEL HISTORIAL
    # -----------------------------------------------------------------------

    def consultar_historial(
        self,
        mensaje_usuario: str,
        thread_id: str,
    ) -> Optional[str]:
        """
        Recupera información de conversaciones anteriores sin utilizar
        herramientas externas.
        """

        mensaje = mensaje_usuario.lower().strip()

        palabras_historial = [
            "folio",
            "cuál fue mi folio",
            "cual fue mi folio",
            "qué folio",
            "que folio",
            "recuerdas",
            "recuérdame",
            "recuerdame",
            "qué motivo tenía mi ticket",
            "que motivo tenia mi ticket",
            "mi ticket anterior",
            "el ticket anterior",
            "qué solicitud realicé",
            "que solicitud realice",
            "qué te pedí anteriormente",
            "que te pedi anteriormente",
        ]

        # Verificar si el estudiante pregunta por información anterior.
        consulta_historial = any(
            palabra in mensaje for palabra in palabras_historial
        )

        if not consulta_historial:
            return None

        config = {"configurable": {"thread_id": thread_id}}

        try:
            estado = self.agente.get_state(config)

            if not estado or not estado.values:
                return None

            mensajes = estado.values.get("messages", [])

            if not mensajes:
                return None

            # Buscar el folio en los mensajes anteriores.
            for indice in range(len(mensajes) - 1, -1, -1):

                contenido = str(mensajes[indice].content)

                coincidencia = re.search(
                    r"TCK-\d+",
                    contenido,
                )

                if not coincidencia:
                    continue

                folio = coincidencia.group(0)

                motivo = (
                    "la solicitud de atención realizada anteriormente"
                )

                # Buscar la solicitud original del estudiante.
                for mensaje_anterior in reversed(mensajes[:indice]):

                    if mensaje_anterior.type != "human":
                        continue

                    texto = str(mensaje_anterior.content)

                    texto = re.sub(
                        r"\[RUT[^\]]*\]\s*",
                        "",
                        texto,
                    ).strip()

                    texto_lower = texto.lower()

                    if (
                        "ticket" in texto_lower
                        or "apelar" in texto_lower
                    ):
                        motivo = texto
                        break

                logger.info(
                    "Folio recuperado desde el historial: %s",
                    folio,
                )

                return (
                    f"El número de folio de tu ticket es {folio}.\n"
                    f"El motivo de la solicitud era: {motivo}"
                )

            logger.info(
                "No se encontró ningún folio en el historial de la sesión."
            )

            return (
                "No encontré un número de folio en el historial "
                "de esta sesión."
            )

        except Exception as exc:

            logger.error(
                "Error consultando el historial: %s",
                exc,
            )

            return None

    # -----------------------------------------------------------------------
    # PROCESAMIENTO DE MENSAJES
    # -----------------------------------------------------------------------

    def responder(
        self,
        mensaje_usuario: str,
        thread_id: str,
        rut_estudiante: Optional[str] = None,
    ) -> str:
        """Procesa una consulta del usuario mediante LangGraph."""

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        try:

            # 1. Revisar primero el historial de la conversación.
            respuesta_historial = self.consultar_historial(
                mensaje_usuario,
                thread_id,
            )

            if respuesta_historial is not None:
                return respuesta_historial

            # 2. Preparar el contexto del estudiante.
            contexto_entrada = mensaje_usuario

            if rut_estudiante and "[RUT" not in mensaje_usuario:
                contexto_entrada = (
                    f"[RUT de la sesión actual: {rut_estudiante}]\n"
                    f"{mensaje_usuario}"
                )

            # 3. Ejecutar el agente LangGraph.
            resultado = self.agente.invoke(
                {
                    "messages": [
                        ("user", contexto_entrada)
                    ]
                },
                config=config,
            )

            # 4. Obtener la respuesta final.
            ultimo_mensaje = resultado["messages"][-1]

            texto_respuesta = str(
                ultimo_mensaje.content
            ).strip()

            # 5. Evitar mostrar llamadas técnicas como respuestas.
            if not texto_respuesta:
                logger.error(
                    "El agente devolvió una respuesta vacía."
                )

                return (
                    "No pude generar una respuesta. "
                    "Por favor, intenta nuevamente."
                )

            if (
                '"parameters"' in texto_respuesta
                and '"name"' in texto_respuesta
            ):
                logger.error(
                    "El modelo devolvió una llamada de herramienta como texto."
                )

                return (
                    "No pude procesar correctamente tu consulta. "
                    "Por favor, intenta nuevamente."
                )

            # 6. Limpiar etiquetas de contexto del RUT.
            if texto_respuesta.startswith("[RUT"):

                partes = texto_respuesta.split("\n", 1)

                if len(partes) > 1:
                    texto_respuesta = partes[1].strip()

            return texto_respuesta

        except Exception as exc:

            logger.error(
                "Error durante la ejecución del agente: %s",
                exc,
            )

            return (
                f"Ocurrió un error al procesar la solicitud: {exc}"
            )

    # -----------------------------------------------------------------------
    # MÉTODO DE ENTRADA
    # -----------------------------------------------------------------------

    def ejecutar(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Método de entrada desde src/main.py."""

        mensaje_usuario = None

        if args:
            mensaje_usuario = args[0]

        else:
            for clave in [
                "mensaje_usuario",
                "mensaje",
                "consulta",
                "pregunta",
                "prompt",
                "user_input",
                "entrada",
                "texto",
            ]:

                if clave in kwargs:
                    mensaje_usuario = kwargs[clave]
                    break

        # Identificador de sesión.
        thread_id = "default_session"

        if len(args) > 1:
            thread_id = str(args[1])

        else:
            for clave in [
                "thread_id",
                "sesion_id",
                "session_id",
                "id_sesion",
            ]:

                if clave in kwargs:
                    thread_id = str(kwargs[clave])
                    break

        # RUT del estudiante.
        rut_estudiante = None

        if len(args) > 2:
            rut_estudiante = args[2]

        else:
            for clave in [
                "rut_estudiante",
                "rut",
                "rut_usuario",
            ]:

                if clave in kwargs:
                    rut_estudiante = kwargs[clave]
                    break

        if mensaje_usuario is None:
            mensaje_usuario = str(kwargs)

        # Procesar solicitud.
        texto_respuesta = self.responder(
            mensaje_usuario=str(mensaje_usuario),
            thread_id=thread_id,
            rut_estudiante=rut_estudiante,
        )

        return {
            "respuesta": texto_respuesta,
            "mensaje": texto_respuesta,
        }
