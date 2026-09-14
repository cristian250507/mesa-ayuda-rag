"""Configuración central del sistema RAG.

Centraliza parámetros de conexión, chunking, recuperación y generación para
evitar valores mágicos dispersos en el código (trazabilidad y mantenibilidad).

Asignatura: ISY0101 - Ingeniería de Soluciones con IA
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
RAIZ: Path = Path(__file__).resolve().parent.parent
DIR_DATA: Path = RAIZ / "data"
DIR_INTERNO: Path = DIR_DATA / "interno"
DIR_EXTERNO: Path = DIR_DATA / "externo"
DIR_INDICE: Path = RAIZ / "indice_faiss"


@dataclass(frozen=True)
class ConfigModelo:
    """Parámetros de conexión y generación del LLM (GitHub Models)."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "GITHUB_BASE_URL", "https://models.inference.ai.azure.com"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    modelo_chat: str = "gpt-4o-mini"
    modelo_embeddings: str = "text-embedding-3-small"

    # temperature baja => respuestas deterministas, menor tasa de alucinación
    temperature: float = 0.1
    max_tokens: int = 700
    timeout: int = 60


@dataclass(frozen=True)
class ConfigRAG:
    """Parámetros del pipeline de recuperación aumentada."""

    # Chunking: equilibrio entre perder contexto (muy chico) y diluir
    # el significado semántico (muy grande). Overlap evita cortar artículos.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Recuperación
    top_k: int = 4
    # Umbral de distancia L2 de FAISS: a MENOR valor, mayor similitud.
    # Si ningún chunk baja del umbral, no se invoca al generador.
    umbral_distancia: float = 1.15


@dataclass(frozen=True)
class ConfigMemoria:
    """Parámetros de la memoria conversacional."""

    ventana_turnos: int = 5          # ConversationBufferWindowMemory(k=5)
    umbral_resumen: int = 10         # a partir de aquí se resume la sesión


CONFIG_MODELO = ConfigModelo()
CONFIG_RAG = ConfigRAG()
CONFIG_MEMORIA = ConfigMemoria()


def validar_entorno() -> None:
    """Verifica que las variables de entorno obligatorias estén definidas.

    Raises:
        EnvironmentError: si falta GITHUB_TOKEN.
    """
    if not CONFIG_MODELO.api_key:
        raise EnvironmentError(
            "Falta la variable de entorno GITHUB_TOKEN. "
            "Defínela antes de ejecutar (ver README.md)."
        )
