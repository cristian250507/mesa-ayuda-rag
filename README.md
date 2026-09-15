# Mesa de Ayuda Académica Inteligente — Agente LLM + RAG

Sistema de recuperación aumentada (RAG) que responde consultas sobre normativa académica y beneficios estudiantiles utilizando una base documental controlada.

El sistema recupera información relevante desde documentos institucionales y normativos, utiliza un modelo de lenguaje local para generar las respuestas y muestra las fuentes documentales utilizadas. Cuando la información recuperada no es suficiente, el sistema puede recomendar la derivación a atención humana.

**Asignatura:** ISY0101 — Ingeniería de Soluciones con IA

**Evaluación:** Parcial 1 (30%) — Diseño de solución con LLM y RAG

---

## 1. Descripción del problema

La Dirección de Asuntos Estudiantiles recibe un volumen alto y estacional de consultas repetitivas relacionadas con normativa académica, beneficios estudiantiles, plazos y procedimientos.

Las respuestas pueden variar entre distintos canales de atención y un error en la interpretación de una normativa puede afectar directamente a un estudiante.

Este proyecto propone una solución basada en **LLM + RAG** para automatizar la primera línea de atención sobre una base documental controlada.

El sistema permite:

- Recuperar información relevante desde documentos institucionales.
- Utilizar documentos internos y externos.
- Generar respuestas utilizando el contexto documental recuperado.
- Mostrar las fuentes utilizadas para responder.
- Mantener contexto entre preguntas de una misma conversación.
- Reformular preguntas de seguimiento para mejorar la recuperación.
- Recomendar derivación a un ejecutivo cuando corresponde.
- Evaluar la calidad de la recuperación y de las respuestas generadas.

---

## 2. Arquitectura

![Arquitectura de la solución](docs/arquitectura.svg)

El pipeline sigue el ciclo:

**Retrieve → Augment → Generate**

y se divide en tres fases principales:

| Fase | Módulos | Descripción |
|---|---|---|
| **A. Indexación (offline)** | Ingesta → Chunking → Embeddings → FAISS | Carga los documentos, los fragmenta, genera embeddings y construye el índice vectorial. |
| **B. Consulta (runtime)** | Memoria → Reformulación → Retrieve → Augment → Generate → Post-proceso | Recibe la consulta, recupera información relevante y genera una respuesta basada en el contexto documental. |
| **C. Evaluación** | Métricas + conjunto dorado | Evalúa la calidad de la recuperación y de la generación. |

### Flujo general

```text
                    DOCUMENTOS
                        │
                        ▼
                  ┌───────────┐
                  │  Ingesta  │
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │  Chunking │
                  └─────┬─────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Ollama             │
              │ nomic-embed-text   │
              └─────────┬──────────┘
                        │
                        ▼
                  ┌───────────┐
                  │   FAISS   │
                  └─────┬─────┘
                        │
                        │
                 CONSULTA USUARIO
                        │
                        ▼
                  ┌───────────┐
                  │  Memoria  │
                  └─────┬─────┘
                        │
                        ▼
                  ┌──────────────┐
                  │ Reformulación│
                  └──────┬───────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Retrieval  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Contexto  │
                  └──────┬──────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Ollama         │
                 │ Gemma 3 4B     │
                 └───────┬────────┘
                         │
                         ▼
                    RESPUESTA
```

---

## 3. Fuentes de información

El sistema utiliza dos tipos de fuentes documentales:

| Tipo | Ubicación | Documentos |
|---|---|---|
| **Internas** | `data/interno/` | Reglamento Académico, Calendario Académico y FAQ de la Mesa de Ayuda |
| **Externas** | `data/externo/` | Normativa de Beneficios Estudiantiles y Marco de Reconocimiento de Aprendizajes |

Ambos tipos de documentos utilizan el mismo pipeline de procesamiento e indexación.

Cada documento mantiene metadatos que permiten identificar su origen y distinguir entre información interna y externa.

---

## 4. Estructura del repositorio

```text
mesa-ayuda-rag/
│
├── config/
│   └── settings.py              # Configuración central del sistema
│
├── data/
│   ├── interno/                 # Corpus institucional
│   └── externo/                 # Corpus normativo externo
│
├── docs/
│   ├── arquitectura.svg         # Diagrama de arquitectura
│   ├── Informe_Tecnico.docx     # Informe técnico
│   └── Guion_Defensa.md         # Guion de defensa
│
├── evaluacion/
│   └── conjunto_dorado.json     # Casos de evaluación
│
├── indice_faiss/                # Índice vectorial generado
│
├── src/
│   ├── ingesta.py               # Carga y fragmentación de documentos
│   ├── vector_store.py          # Embeddings, FAISS y recuperación
│   ├── prompts.py               # Prompts del sistema
│   ├── memoria.py               # Memoria conversacional
│   ├── rag_pipeline.py          # Pipeline RAG y generación
│   ├── evaluacion.py            # Métricas de evaluación
│   └── main.py                  # Punto de entrada CLI
│
├── .env.example                 # Ejemplo de configuración
├── requirements.txt             # Dependencias Python
└── README.md
```

---

## 5. Tecnologías utilizadas

### Lenguaje

- Python 3.10 o superior.

### Inteligencia artificial

- **Ollama** — ejecución local de modelos de inteligencia artificial.
- **Gemma 3 4B** — generación de respuestas.
- **nomic-embed-text** — generación de embeddings.

### RAG y búsqueda vectorial

- **LangChain** — integración de los componentes del pipeline RAG.
- **FAISS** — almacenamiento y búsqueda vectorial.
- **Embeddings locales** — representación semántica de los documentos.

### Procesamiento documental

- `pypdf`
- Python estándar.

### Evaluación

- Context Precision.
- Context Recall.
- Faithfulness.
- Answer Relevancy.

---

## 6. Requisitos previos

Antes de ejecutar el proyecto se necesita:

- Python 3.10 o superior.
- Ollama instalado.
- Modelo `gemma3:4b`.
- Modelo `nomic-embed-text`.
- Dependencias Python del proyecto.

### Verificar Ollama

Para comprobar que Ollama está instalado:

```bash
ollama --version
```

Los modelos requeridos pueden descargarse con:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

En Windows, si `ollama` no está disponible directamente en el PATH, se puede utilizar su ejecutable instalado localmente:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull gemma3:4b
```

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull nomic-embed-text
```

Para comprobar que el modelo de generación funciona:

```powershell
ollama run gemma3:4b
```

---

## 7. Instalación

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/<usuario>/mesa-ayuda-rag.git
cd mesa-ayuda-rag
```

### Paso 2 — Crear el entorno virtual

```bash
python -m venv .venv
```

### Paso 3 — Activar el entorno virtual

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Paso 4 — Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## 8. Configuración

El proyecto utiliza un archivo `.env` para definir la configuración de Ollama.

Crear un archivo `.env` en la raíz del proyecto:

```env
# ============================================================
# Ollama - ejecución local
# ============================================================

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

También se incluye `.env.example` como referencia.

El archivo `.env` está destinado a la configuración local y no debe subirse al repositorio.

No se requieren claves de API de OpenAI ni de GitHub para ejecutar el sistema.

---

## 9. Ejecución

### Paso 1 — Construir el índice vectorial

La primera vez que se ejecuta el proyecto es necesario procesar los documentos:

```bash
python -m src.main --indexar
```

Este proceso:

1. Carga los documentos desde `data/`.
2. Divide los documentos en chunks.
3. Genera embeddings mediante `nomic-embed-text`.
4. Construye el índice FAISS.
5. Guarda el índice en `indice_faiss/`.

La indexación debe repetirse cuando se agreguen, eliminen o modifiquen documentos del corpus.

---

### Paso 2 — Iniciar la atención conversacional

```bash
python -m src.main --chat
```

El sistema permite realizar consultas directamente desde la consola.

Comandos disponibles:

```text
reiniciar
```

Reinicia el contexto de la conversación.

```text
salir
```

Finaliza la aplicación.

---

### Paso 3 — Evaluar el sistema

```bash
python -m src.main --evaluar
```

Este comando ejecuta los casos definidos en:

```text
evaluacion/conjunto_dorado.json
```

y calcula métricas relacionadas con:

- **Context Precision**
- **Context Recall**
- **Faithfulness**
- **Answer Relevancy**

---

## 10. Ejemplo de uso

```text
Estudiante > Reprobé dos veces Cálculo, ¿qué me pasa?

Asistente > Reprobar por segunda vez consecutiva una misma asignatura
constituye una causal de eliminación académica según el Reglamento Académico.

Fuentes: Reglamento_Academico (interno)
Confianza: alto | Chunks usados: 3
```

Luego el estudiante puede realizar una pregunta relacionada:

```text
Estudiante > ¿Y eso afecta mi beca?
```

El sistema utiliza el contexto de la conversación para interpretar que "eso" hace referencia a la situación académica mencionada anteriormente.

La consulta se reformula internamente antes de realizar la búsqueda vectorial, permitiendo recuperar información relacionada con beneficios estudiantiles.

```text
Asistente > Según la normativa recuperada, la situación puede afectar la
mantención del beneficio. Se recomienda revisar las condiciones específicas
del beneficio y, si corresponde, consultar con la Dirección de Asuntos
Estudiantiles.

Fuentes: Normativa_Beneficios_Estudiantiles (externo)
Confianza: alto | Chunks usados: 2
```

Este ejemplo demuestra la **gestión de contexto conversacional**, ya que una pregunta dependiente del contexto anterior puede ser reformulada antes de realizar la recuperación de información.

---

## 11. Parámetros configurables

Los principales parámetros se encuentran centralizados en:

```text
config/settings.py
```

| Parámetro | Valor | Justificación |
|---|---:|---|
| `chunk_size` | 800 | Equilibrio entre conservar contexto y evitar chunks demasiado grandes. |
| `chunk_overlap` | 120 | Permite mantener continuidad entre fragmentos. |
| `top_k` | 4 | Recupera los documentos más relevantes sin saturar el contexto. |
| `umbral_distancia` | 1.15 | Filtra resultados con baja similitud antes de generar una respuesta. |
| `temperature` | 0.1 | Favorece respuestas más deterministas para información normativa. |
| `ventana_turnos` | 5 | Mantiene los últimos turnos de la conversación. |
| `umbral_resumen` | 10 | Define cuándo se intenta generar un resumen de la conversación. |

---

## 12. Componentes principales

### Ingesta y Chunking

`src/ingesta.py`

Carga los documentos del corpus y los divide en fragmentos de tamaño controlado para facilitar la búsqueda semántica.

### Embeddings y FAISS

`src/vector_store.py`

Utiliza `nomic-embed-text` mediante Ollama para transformar los chunks en vectores y FAISS para realizar la recuperación de información relevante.

### Memoria

`src/memoria.py`

Mantiene el contexto de la conversación para permitir preguntas de seguimiento y reformulación de consultas.

### Pipeline RAG

`src/rag_pipeline.py`

Coordina:

1. Memoria conversacional.
2. Reformulación de la consulta.
3. Recuperación de documentos.
4. Construcción del contexto.
5. Generación mediante Gemma 3 4B.
6. Post-procesamiento de la respuesta.
7. Registro de fuentes y nivel de confianza.

### Evaluación

`src/evaluacion.py`

Permite medir por separado la calidad de la recuperación y la calidad de las respuestas generadas.

---

## 13. Flujo de una consulta

Una consulta del estudiante sigue el siguiente proceso:

```text
Pregunta del estudiante
          │
          ▼
Memoria conversacional
          │
          ▼
Reformulación de la consulta
          │
          ▼
Búsqueda semántica en FAISS
          │
          ▼
Filtrado por umbral de similitud
          │
          ▼
Construcción del contexto
          │
          ▼
Gemma 3 4B
          │
          ▼
Post-procesamiento
          │
          ▼
Respuesta + fuentes + confianza
```

Este flujo permite separar claramente la **recuperación de información** de la
**generación de lenguaje**, reduciendo el riesgo de que el modelo responda sin
respaldo documental.

---

## 14. Ventajas de la solución

La arquitectura propuesta permite:

- Ejecutar el sistema de forma local.
- No depender de una API externa de pago para la generación ni los embeddings.
- Mantener control sobre los documentos utilizados como fuente.
- Incorporar trazabilidad mediante las fuentes recuperadas.
- Mantener contexto conversacional.
- Separar recuperación y generación.
- Utilizar modelos diferentes para generación y embeddings.
- Evaluar el comportamiento del sistema mediante un conjunto de pruebas.
- Mantener una arquitectura modular que permite reemplazar componentes en el futuro.

---

## 15. Limitaciones conocidas

- El sistema **no reemplaza la atención de un funcionario o ejecutivo**.
- No resuelve casos particulares que requieran interpretación administrativa o jurídica.
- La calidad de las respuestas depende directamente de la calidad y vigencia de los documentos almacenados en `data/`.
- Un documento desactualizado puede producir una respuesta correcta respecto al corpus, pero incorrecta respecto a la normativa vigente.
- El modelo se ejecuta localmente, por lo que el tiempo de respuesta depende del hardware disponible.
- Las métricas de generación basadas en evaluación automática pueden presentar variabilidad y deben interpretarse como indicadores de calidad, no como una medición absoluta.

---

## 16. Seguridad y configuración

El sistema utiliza modelos locales mediante Ollama y no requiere claves de API externas.

La comunicación con Ollama se realiza mediante el servidor local:

```text
http://localhost:11434
```

Los modelos utilizados son:

```text
Generación:
Gemma 3 4B

Embeddings:
nomic-embed-text
```

El archivo `.env` contiene únicamente parámetros de configuración local y debe mantenerse fuera del repositorio.

---

## 17. Autores

Proyecto desarrollado en modalidad de trabajo en parejas para la Evaluación Parcial 1 de:

**ISY0101 — Ingeniería de Soluciones con IA**