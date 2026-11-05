"""Módulo de Memoria de Corto y Largo Plazo del Agente.

Integra la memoria conversacional de corto plazo (LangGraph Checkpointer)
y la memoria de largo plazo (persistencia JSON por estudiante).

Asignatura: ISY0101 - Ingeniería de Soluciones con IA (EP2)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver

from config.settings import DIR_DATA

logger = logging.getLogger(__name__)

RUTA_MEMORIA_LARGO_PLAZO = DIR_DATA / "memoria_estudiantes.json"


class MemoriaLargoPlazo:
    """Gestiona el perfil persistente de los estudiantes a largo plazo."""

    def __init__(self) -> None:
        self._cargar_memoria()

    def _cargar_memoria(self) -> None:
        if RUTA_MEMORIA_LARGO_PLAZO.exists():
            try:
                with open(RUTA_MEMORIA_LARGO_PLAZO, "r", encoding="utf-8") as f:
                    self._datos = json.load(f)
            except json.JSONDecodeError:
                self._datos = {}
        else:
            self._datos = {}

    def guardar_memoria(self) -> None:
        """Persiste los datos de largo plazo en disco."""
        DIR_DATA.mkdir(parents=True, exist_ok=True)
        with open(RUTA_MEMORIA_LARGO_PLAZO, "w", encoding="utf-8") as f:
            json.dump(self._datos, f, ensure_ascii=False, indent=2)

    def obtener_perfil(self, rut: str) -> Optional[Dict[str, Any]]:
        """Recupera el historial o perfil de un estudiante por su RUT."""
        return self._datos.get(rut)

    def actualizar_perfil(self, rut: str, datos: Dict[str, Any]) -> None:
        """Guarda o actualiza preferencias/datos persistentes de un estudiante."""
        if rut not in self._datos:
            self._datos[rut] = {}
        self._datos[rut].update(datos)
        self.guardar_memoria()
        logger.info("Memoria a largo plazo actualizada para el RUT: %s", rut)


def obtener_checkpointer_corto_plazo() -> MemorySaver:
    """Retorna la instancia de memoria de corto plazo para LangGraph."""
    return MemorySaver()