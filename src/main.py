"""Punto de entrada de la Mesa de Ayuda Académica Inteligente.

Modos de uso:
    python -m src.main --indexar     Construye el índice vectorial.
    python -m src.main --chat        Inicia la atención conversacional.
    python -m src.main --evaluar     Ejecuta el ciclo de evaluación del RAG.

El sistema utiliza Ollama localmente para generación y embeddings.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from config.settings import RAIZ
from src.evaluacion import (
    cargar_conjunto_dorado,
    context_precision,
    context_recall,
    evaluar_generacion,
    resumir,
)
from src.ingesta import cargar_corpus, fragmentar
from src.rag_pipeline import AgenteNormativo, crear_llm
from src.vector_store import (
    cargar_indice,
    construir_indice,
    formatear_contexto,
    recuperar,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")


def comando_indexar() -> None:
    """Ejecuta la ingesta, el chunking y la construcción del índice FAISS."""
    logger.info(
        "Iniciando indexación del corpus..."
    )

    documentos = cargar_corpus()

    chunks = fragmentar(
        documentos
    )

    construir_indice(
        chunks
    )

    logger.info(
        "Indexación finalizada correctamente."
    )


def comando_chat() -> None:
    """Inicia el bucle conversacional en consola."""
    agente = AgenteNormativo(
        cargar_indice()
    )

    print("\n" + "=" * 68)
    print(
        " MESA DE AYUDA ACADÉMICA INTELIGENTE"
    )
    print(
        " Escribe 'salir' para terminar o "
        "'reiniciar' para una nueva atención."
    )
    print("=" * 68 + "\n")

    while True:
        try:
            pregunta = input(
                "Estudiante > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\nAtención finalizada."
            )
            break

        if pregunta.lower() in {
            "salir",
            "exit",
            "quit",
        }:
            print(
                "Atención finalizada."
            )
            break

        if pregunta.lower() == "reiniciar":
            agente.reiniciar()

            print(
                "Sesión reiniciada.\n"
            )

            continue

        respuesta, evidencia = agente.responder(
            pregunta
        )

        print(
            f"\nAsistente > "
            f"{respuesta['respuesta']}"
        )

        if respuesta["fuentes"]:
            citas = ", ".join(
                f"{f['documento']} ({f['tipo']})"
                for f in respuesta["fuentes"]
            )

            print(
                f"  Fuentes: {citas}"
            )

        print(
            f"  Confianza: "
            f"{respuesta['nivel_confianza']}",
            end="",
        )

        if respuesta.get(
            "derivar_a_humano"
        ):
            print(
                " | Se recomienda derivar "
                "a un ejecutivo.",
                end="",
            )

        print(
            f" | Chunks usados: "
            f"{len(evidencia)}\n"
        )


def comando_evaluar() -> None:
    """Ejecuta el conjunto dorado y reporta las métricas del sistema."""
    casos = cargar_conjunto_dorado(
        RAIZ
        / "evaluacion"
        / "conjunto_dorado.json"
    )

    if not casos:
        logger.error(
            "No hay casos de evaluación disponibles."
        )
        return

    indice = cargar_indice()

    agente = AgenteNormativo(
        indice
    )

    juez = crear_llm()

    resultados: List[dict] = []

    for numero, caso in enumerate(
        casos,
        start=1,
    ):
        pregunta: str = caso[
            "pregunta"
        ]

        relevantes: List[str] = caso.get(
            "documentos_relevantes",
            [],
        )

        recuperados = [
            doc
            for doc, _ in recuperar(
                indice,
                pregunta,
            )
        ]

        contexto = formatear_contexto(
            [
                (doc, 0.0)
                for doc in recuperados
            ]
        )

        respuesta, _ = agente.responder(
            pregunta
        )

        agente.reiniciar()

        metricas = {
            "context_precision": context_precision(
                recuperados,
                relevantes,
            ),
            "context_recall": context_recall(
                recuperados,
                relevantes,
            ),
            **evaluar_generacion(
                juez,
                pregunta,
                respuesta["respuesta"],
                contexto,
            ),
        }

        resultados.append(
            metricas
        )

        logger.info(
            "Caso %d/%d evaluado: %s",
            numero,
            len(casos),
            metricas,
        )

    print("\n" + "=" * 68)
    print(
        " RESULTADOS GLOBALES DE EVALUACIÓN"
    )
    print("=" * 68)

    for metrica, valor in resumir(
        resultados
    ).items():
        print(
            f"  {metrica:.<40} {valor}"
        )

    print()


def main() -> int:
    """Procesa los argumentos de línea de comandos.

    Returns:
        Código de salida del proceso (0 = éxito).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mesa de Ayuda Académica "
            "Inteligente (LLM + RAG)."
        )
    )

    grupo = parser.add_mutually_exclusive_group(
        required=True
    )

    grupo.add_argument(
        "--indexar",
        action="store_true",
        help="Construir el índice.",
    )

    grupo.add_argument(
        "--chat",
        action="store_true",
        help="Iniciar la atención.",
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
        logger.error(
            "Recurso no encontrado: %s",
            exc,
        )

        return 1

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Error inesperado: %s",
            exc,
        )

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())