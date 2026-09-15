"""Configuración central del sistema RAG.

Centraliza parámetros de conexión, chunking, recuperación y generación para
evitar valores mágicos dispersos en el código (trazabilidad y mantenibilidad).

El sistema utiliza Ollama de forma local, por lo que no requiere una API
externa ni una clave de OpenAI.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #

RAIZ: Path = Path(__file__).resolve().parent.parent

DIR_DATA: Path = RAIZ / "data"
DIR_INTERNO: Path = DIR_DATA / "interno"
DIR_EXTERNO: Path = DIR_DATA / "externo"
DIR_INDICE: Path = RAIZ / "indice_faiss"


# --------------------------------------------------------------------------- #
# Variables de entorno
# --------------------------------------------------------------------------- #

load_dotenv(RAIZ / ".env")


# --------------------------------------------------------------------------- #
# Configuración de modelos
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigModelo:
    """Parámetros de los modelos locales de Ollama."""

    # Servidor local de Ollama.
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    # Modelo utilizado para generar y reformular respuestas.
    modelo_chat: str = os.getenv(
        "OLLAMA_CHAT_MODEL",
        "gemma3:4b",
    )

    # Modelo utilizado para generar embeddings.
    modelo_embeddings: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text",
    )

    # Temperature baja => respuestas más deterministas.
    temperature: float = 0.1

    # Límite conceptual de generación.
    max_tokens: int = 350

    # Tiempo máximo de espera de conexión.
    timeout: int = 60


# --------------------------------------------------------------------------- #
# Configuración RAG
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigRAG:
    """Parámetros del pipeline de recuperación aumentada."""

    # Chunking:
    # equilibrio entre perder contexto (muy chico) y diluir
    # el significado semántico (muy grande).
    chunk_size: int = 800

    chunk_overlap: int = 120

    # Número máximo de chunks recuperados.
    top_k: int = 4

    # Umbral de distancia L2 de FAISS:
    # a MENOR valor, mayor similitud.
    #
    # Si ningún chunk baja del umbral,
    # no se invoca al generador.
    umbral_distancia: float = 1.15


# --------------------------------------------------------------------------- #
# Configuración de memoria
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigMemoria:
    """Parámetros de la memoria conversacional."""

    # Cantidad de turnos recientes que se mantienen.
    ventana_turnos: int = 5

    # Cantidad de turnos antes de generar un resumen.
    umbral_resumen: int = 10


# --------------------------------------------------------------------------- #
# Instancias globales de configuración
# --------------------------------------------------------------------------- #

CONFIG_MODELO = ConfigModelo()

CONFIG_RAG = ConfigRAG()

CONFIG_MEMORIA = ConfigMemoria()