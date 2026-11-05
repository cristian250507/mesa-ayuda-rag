"""Configuración global del sistema RAG / Agente.

Asignatura: ISY0101 - Ingeniería de Soluciones con IA (EP2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env si existe
load_dotenv()

# Directorios principales del proyecto
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATA = RAIZ / "data"
DIR_INDICE = RAIZ / "indice_faiss"
DIR_EXTERNO = DIR_DATA / "corpus_externo"
DIR_INTERNO = DIR_DATA / "corpus_interno"


@dataclass(frozen=True)
class ConfiguracionModelo:
    """Configuración del LLM y Embeddings locales (Ollama)."""

    modelo_chat: str = os.getenv("MODELO_CHAT", "llama3.2")
    modelo_embeddings: str = os.getenv("MODELO_EMBEDDINGS", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = 0.0
    k_documentos: int = 4
    top_k: int = 4
    tamano_chunk: int = 500
    solapamiento: int = 50
    umbral_distancia: float = float(os.getenv("UMBRAL_DISTANCIA", "0.7"))


@dataclass(frozen=True)
class ConfiguracionChunking:
    """Parámetros de fragmentación del corpus."""

    tamano_chunk: int = 500
    solapamiento: int = 50


@dataclass(frozen=True)
class ConfiguracionRecuperacion:
    """Parámetros de recuperación de información."""

    k_documentos: int = 4
    top_k: int = 4
    umbral_distancia: float = 0.7


@dataclass(frozen=True)
class ConfiguracionMemoria:
    """Parámetros de la memoria conversacional."""

    ventana_turnos: int = 5
    umbral_resumen: int = 3


# Instancias globales de configuración
CONFIG_MODELO = ConfiguracionModelo()
CONFIG_CHUNKING = ConfiguracionChunking()
CONFIG_RECUPERACION = ConfiguracionRecuperacion()
CONFIG_MEMORIA = ConfiguracionMemoria()

# Alias de compatibilidad para ingesta.py, vector_store.py y evaluacion.py
CONFIG_RAG = CONFIG_MODELO