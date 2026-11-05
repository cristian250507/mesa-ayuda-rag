"""Punto de entrada de la Mesa de Ayuda Académica Inteligente (Agente LangGraph).

Modos de uso:
    python -m src.main --indexar     Construye o actualiza el índice FAISS.
    python -m src.main --chat        Inicia la atención conversacional con el Agente.
    python -m src.main --evaluar     Ejecuta pruebas sobre el agente orquestado.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA (EP2)
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from config.settings import RAIZ
from src.agente import AgenteOrquestador
from src.evaluacion import cargar_conjunto_dorado
from src.ingesta import cargar_corpus, fragmentar
from src.vector_store import construir_indice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")


def comando_indexar() -> None:
    """Ejecuta la ingesta, el chunking y la construcción del índice FAISS."""
    logger.info("Iniciando indexación del corpus...")
    documentos = cargar_corpus()
    chunks = fragmentar(documentos)
    construir_indice(chunks)
    logger.info("Indexación finalizada correctamente.")


def comando_chat() -> None:
    """Inicia el bucle conversacional del Agente en consola."""
    agente = AgenteOrquestador()
    thread_id = f"sesion_{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 68)
    print(" MESA DE AYUDA ACADÉMICA INTELIGENTE - AGENTE LANGGRAPH (EP2)")
    print(" Escribe 'salir' para terminar o 'reiniciar' para una nueva sesión.")
    print(f" ID de Sesión (Memoria Corto Plazo): {thread_id}")
    print("=" * 68 + "\n")

    rut_estudiante = input(
        "Estudiante > Ingresa tu RUT (opcional para memoria a largo plazo, o presiona Enter): "
    ).strip()

    if not rut_estudiante:
        rut_estudiante = None
    else:
        print(f"✓ Sesión vinculada al RUT: {rut_estudiante}\n")

    while True:
        try:
            pregunta = input("Estudiante > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAtención finalizada.")
            break

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("Atención finalizada.")
            break

        if pregunta.lower() == "reiniciar":
            thread_id = f"sesion_{uuid.uuid4().hex[:8]}"
            print(f"\n✓ Sesión reiniciada. Nueva ID de sesión: {thread_id}\n")
            continue

        # Ejecución del agente con orquestación de herramientas y memoria
        resultado = agente.ejecutar(
            entrada_usuario=pregunta,
            thread_id=thread_id,
            rut_estudiante=rut_estudiante,
        )

        print(f"\nAgente > {resultado['respuesta']}\n")


def comando_evaluar() -> None:
    """Ejecuta el conjunto de prueba para evaluar la orquestación del Agente."""
    casos = cargar_conjunto_dorado(
        RAIZ / "evaluacion" / "conjunto_dorado.json"
    )
    if not casos:
        logger.error("No hay casos de evaluación disponibles.")
        return

    agente = AgenteOrquestador()
    logger.info("Iniciando evaluación de casos con el Agente Orquestador...")

    for numero, caso in enumerate(casos, start=1):
        pregunta = caso["pregunta"]
        thread_eval = f"eval_caso_{numero}"

        logger.info("Evaluando Caso %d/%d: '%s'", numero, len(casos), pregunta)
        resultado = agente.ejecutar(
            entrada_usuario=pregunta,
            thread_id=thread_eval,
        )
        print(f"Caso {numero} | Pregunta: {pregunta}")
        print(f"Respuesta del Agente: {resultado['respuesta']}\n" + "-" * 50)


def main() -> int:
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Mesa de Ayuda Académica Inteligente - Agente LangGraph (EP2)."
    )

    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--indexar",
        action="store_true",
        help="Construir o actualizar el índice FAISS.",
    )
    grupo.add_argument(
        "--chat",
        action="store_true",
        help="Iniciar la atención conversacional.",
    )
    grupo.add_argument(
        "--evaluar",
        action="store_true",
        help="Evaluar el sistema.",
    )

    argumentos = parser.parse_args()

    try:
        if argumentos.indexar:
            comando_indexar()
        elif argumentos.chat:
            comando_chat()
        else:
            comando_evaluar()
    except FileNotFoundError as exc:
        logger.error("Recurso no encontrado: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())