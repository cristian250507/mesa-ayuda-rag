# Mesa de Ayuda Académica Inteligente — Agente Agentico ReAct (EP2)

Sistema de **Agente Inteligente Autónomo** basado en la arquitectura **ReAct (Reasoning + Acting)** orquestado con **LangGraph** y modelos locales mediante **Ollama**, diseñado para la resolución de consultas normativas, evaluación de causales académicas y generación persistente de tickets de atención.

El sistema utiliza **`llama3.2`** como motor del agente, integra búsqueda semántica mediante **FAISS**, ejecuta reglas lógicas de negocio y realiza persistencia de datos I/O en disco (`data/tickets.json`).

---

# 1. Descripción del problema

La **Dirección de Asuntos Estudiantiles** requiere evolucionar la primera línea de atención de un sistema RAG pasivo a un **Agente Agentico Autónomo** capaz no solo de responder dudas, sino de tomar decisiones de negocio y realizar acciones administrativas como:

- Consultar la normativa académica mediante búsqueda vectorial.
- Evaluar de forma autónoma la situación académica de un estudiante (causales de eliminación).
- Crear tickets de atención formal y registrarlos en disco cuando se solicite una apelación.
- Mantener la memoria de la conversación a corto plazo para responder sobre folios o datos entregados previamente.

### Objetivos principales (EP2)

El sistema permite:

- **Razonamiento y Selección de Herramientas:** Decidir autónomamente qué herramienta invocar según la necesidad del estudiante.
- **RAG Vectorial:** Recuperar normativas con FAISS y `nomic-embed-text`.
- **Lógica de Negocio:** Evaluar causales de eliminación por porcentaje de reprobación y reincidencia en asignaturas.
- **Persistencia I/O:** Generar folios únicos (`TCK-XXXXXXX`) y guardar tickets en `data/tickets.json`.
- **Memoria Conversacional:** Gestionar el estado de la sesión (`thread_id`) con `MemorySaver` de LangGraph.

---

# 2. Arquitectura de la solución

La arquitectura se basa en el **Loop ReAct (Reasoning + Acting)** orquestado por LangGraph.

| Componente | Tecnología / Módulo | Descripción |
| :--- | :--- | :--- |
| **Orquestador Agentico** | LangGraph + `MemorySaver` | Gestiona el ciclo ReAct y la memoria a corto plazo por `thread_id`. |
| **Modelo LLM** | `llama3.2` (Ollama) | Razona, decide la invocación de herramientas y sintetiza respuestas. |
| **Herramienta RAG** | `consultar_normativa_academica` | Búsqueda semántica en FAISS (`nomic-embed-text`). |
| **Herramienta Lógica** | `evaluar_causal_eliminacion` | Regla de negocio para cálculo de causales de eliminación. |
| **Herramienta I/O** | `crear_ticket_atencion` | Registro de tickets en `data/tickets.json`. |

## Flujo general del Agente

```text
 ┌─────────────────────────────┐
 │    Consulta del Estudiante  │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │     Agente Orquestador      │
 │   (LangGraph + llama3.2)    │
 └──────────────┬──────────────┘
                │
        ¿Requiere herramienta?
        ┌───────┴───────┐
     SÍ │               │ NO (Usa Memoria / Historial)
        ▼               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│    Selección e Invocación    │ │    Respuesta Directa desde   │
│       de Herramienta         │ │    el Historial de Sesión    │
└──────────────┬───────────────┘ └──────────────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐
│  RAG  │  │Lógica │  │ Ticket│
│ FAISS │  │Negocio│  │  I/O  │
└───┬───┘  └───┬───┘  └───┬───┘
    │          │          │
    └──────────┼──────────┘
               │ (Retorno de Observación)
               ▼
 ┌─────────────────────────────┐
 │ Respuesta Final al Usuario  │
 └─────────────────────────────┘
```

# 3. Estructura del repositorio

```text
mesa-ayuda-rag-main/
│
├── config/
│   └── settings.py              # Configuración central del sistema y modelo
│
├── data/
│   ├── corpus_interno/          # Documentación normativa interna
│   ├── corpus_externo/          # Documentación normativa externa
│   ├── memoria_estudiantes.json # Registro simulación de alumnos
│   └── tickets.json             # Persistencia física de tickets de atención
│
├── indice_faiss/                # Base de datos vectorial indexada
│
├── src/
│   ├── agente.py                # Agente Orquestador LangGraph ReAct y Prompt de Control
│   ├── main.py                  # CLI y punto de entrada interactivo
│   ├── tools.py                 # Definición de herramientas (RAG, Lógica e I/O)
│   ├── vector_store.py          # Búsqueda semántica en FAISS
│   └── ingesta.py               # Carga y fragmentación de documentos
│
├── test_demo.py                 # Script de prueba automatizada
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación técnica
```

---

# 4. Tecnologías utilizadas

### Lenguaje y entorno

- **Python 3.10 o superior**

### Inteligencia Artificial Agéntica

| Componente | Tecnología |
|---|---|
| Orquestador agéntico | **LangGraph (ReAct)** |
| Ejecución local de LLM | **Ollama** |
| Modelo de Chat / Agente | **Llama 3.2 (`llama3.2`)** |
| Modelo de Embeddings | **nomic-embed-text** |
| Base vectorial | **FAISS** |
| Persistencia I/O | **JSON local (`tickets.json`)** |

### Herramientas y procesamiento

- **LangGraph:** orquestación del agente mediante el patrón ReAct.
- **Ollama:** ejecución local de los modelos de inteligencia artificial.
- **FAISS:** almacenamiento y búsqueda de información vectorial.
- **JSON:** almacenamiento local de los tickets de atención.
- **Python:** implementación de la lógica de negocio y herramientas del agente.

---

# 5. Requisitos previos

Antes de ejecutar el proyecto se necesita:

- Python 3.10 o superior.
- Ollama instalado y ejecutándose localmente.
- Modelo `llama3.2`.
- Modelo `nomic-embed-text`.
- Dependencias Python del proyecto.

### Instalar Ollama

Descargar e instalar Ollama desde su página oficial:

[Descargar Ollama](https://ollama.com/download)

### Verificar Ollama

```powershell
ollama --version
```

### Descargar los modelos

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Para comprobar que los modelos están disponibles:

```powershell
ollama list
```

---

# 6. Instalación y ejecución

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/cristian250507/Mesa-ayuda-rag.git
cd Mesa-ayuda-rag
```

### Paso 2 — Crear el entorno virtual

```bash
python -m venv .venv
```

### Paso 3 — Activar el entorno virtual

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Paso 4 — Instalar las dependencias

```bash
pip install -r requirements.txt
```

### Paso 5 — Construir el índice vectorial (si es necesario)

```bash
python -m src.main --indexar
```

Este proceso permite preparar la base vectorial FAISS para realizar consultas sobre la normativa académica.

### Paso 6 — Iniciar el chat con el agente

```bash
python -m src.main --chat
```

El sistema inicia una interfaz conversacional por consola que permite interactuar directamente con el agente.

### Comandos disponibles

| Comando | Acción |
|---|---|
| `salir` | Finaliza la aplicación. |

---

# 7. Herramientas integradas (Tools)

El agente dispone de **3 herramientas principales**, definidas en `src/tools.py`, que permiten ejecutar acciones específicas según la consulta del estudiante.

| Herramienta | Tipo | Descripción |
|---|---|---|
| `consultar_normativa_academica` | RAG Vectorial | Realiza búsquedas de similitud en el índice FAISS para responder consultas sobre plazos, fechas y reglas del reglamento académico. |
| `evaluar_causal_eliminacion` | Lógica de negocio | Procesa la cantidad de asignaturas inscritas y reprobadas para evaluar si el estudiante incurre en una causal de eliminación. |
| `crear_ticket_atencion` | Persistencia I/O | Genera un folio único (`TCK-XXXXXXX`) y almacena la solicitud en `data/tickets.json`. |

### Funcionamiento del agente

El agente utiliza el patrón **ReAct (Reasoning and Acting)** mediante LangGraph, lo que le permite:

- Analizar la consulta del estudiante.
- Identificar qué herramienta necesita utilizar.
- Ejecutar la herramienta correspondiente.
- Interpretar los resultados obtenidos.
- Generar una respuesta final en lenguaje natural.

---

# 8. Ejemplo de uso y validación (EP2)

Para validar el funcionamiento del sistema se realizaron distintas pruebas mediante el chat interactivo:

```bash
python -m src.main --chat
```

### Vinculación del RUT de sesión

```text
RUT de sesión: 21902119-4
```

### Prueba 1 — Consulta RAG Vectorial

**Consulta del estudiante:**

```text
Estudiante > ¿Hasta cuándo puedo retirar una asignatura sin que quede reprobada?
```

**Respuesta del agente:**

```text
Agente > Podrás retirar una asignatura sin que quede
reprobada hasta la semana ocho del semestre...
```

Esta prueba permite comprobar la recuperación de información desde la base vectorial FAISS.

### Prueba 2 — Lógica de negocio

**Consulta del estudiante:**

```text
Estudiante > Inscribí 5 ramos este semestre y reprobé 3.
Además reprobé Cálculo por segunda vez consecutiva.
¿En qué situación académica quedo?
```

**Respuesta del agente:**

```text
Agente > Quedas en situación crítica debido a que
reprobaste por segunda vez consecutiva la asignatura
de Cálculo...
```

Esta prueba valida la ejecución de la herramienta `evaluar_causal_eliminacion`.

### Prueba 3 — Persistencia de ticket (I/O)

**Consulta del estudiante:**

```text
Estudiante > Crea un ticket de atención para apelar
mi causal de eliminación por repruebo de Cálculo.
```

**Respuesta del agente:**

```text
Agente > Tu ticket de atención se ha creado con éxito.
El folio de tu ticket es TCK-1793665295...
```

Esta prueba verifica la generación de folios y el almacenamiento de solicitudes en `data/tickets.json`.

### Prueba 4 — Memoria conversacional (Short-Term Memory)

**Consulta del estudiante:**

```text
Estudiante > ¿Me recuerdas cuál fue el número de folio
que me acabas de entregar y para qué motivo era?
```

**Respuesta del agente:**

```text
Agente > El número de folio de tu ticket es
TCK-1793665295.

El motivo de la solicitud era: apelar causal
de eliminación por repruebo de Cálculo.
```

Esta prueba permite comprobar que el agente conserva el contexto de la conversación y puede recuperar información entregada anteriormente.

---

# 9. Limpieza de registros de prueba

Para restablecer la base de datos de tickets antes de realizar una nueva demostración, se puede ejecutar el siguiente comando en PowerShell:

```powershell
Set-Content -Path data/tickets.json -Value "[]"
```

Este comando elimina los registros almacenados y deja el archivo JSON como una lista vacía, permitiendo comenzar nuevamente las pruebas.

> **Importante:** este procedimiento elimina todos los tickets almacenados en el archivo. Se recomienda utilizarlo únicamente durante las pruebas y demostraciones.

---

# Autores

Proyecto desarrollado por:

- **Cristian Pizarro**
- **Marco Avello**
- **Benjamin Fredes**

**Asignatura:** ISY0101 — Ingeniería de Soluciones con IA

**Evaluación:** Evaluación Parcial 2 (EP2)
