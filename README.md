# Mesa de Ayuda Académica Inteligente — Agente LLM + RAG

Sistema de recuperación aumentada (RAG) que responde consultas sobre normativa
académica y beneficios estudiantiles utilizando una base documental controlada,
citando las fuentes utilizadas y recomendando derivación a atención humana cuando
la información recuperada no es suficiente.

**Asignatura:** ISY0101 — Ingeniería de Soluciones con IA  
**Evaluación:** Parcial 1 (30%) — Diseño de solución con LLM y RAG

---

## 1. Descripción del problema

La Dirección de Asuntos Estudiantiles recibe un volumen alto y estacional de
consultas repetitivas relacionadas con normativa académica, beneficios
estudiantiles, plazos y procedimientos.

Las respuestas pueden variar entre distintos canales de atención y un error en la
interpretación de una normativa puede afectar directamente a un estudiante.

Este proyecto propone una solución basada en **LLM + RAG** para automatizar la
primera línea de atención sobre una base documental controlada.

El sistema permite:

- Recuperar información relevante desde documentos institucionales.
- Utilizar documentos internos y externos.
- Generar respuestas utilizando el contexto documental recuperado.
- Mostrar las fuentes utilizadas para responder.
- Mantener contexto entre preguntas de una misma conversación.
- Reformular preguntas de seguimiento para mejorar la recuperación.
- Recomendar derivación a un ejecutivo cuando corresponde.
- Evaluar la calidad de la recuperación y de las respuestas generadas.

## 2. Arquitectura



El pipeline sigue el ciclo **Retrieve → Augment → Generate** y se divide en tres
fases principales:

| Fase | Módulos | Descripción |
|---|---|---|
| **A. Indexación (offline)** | Ingesta → Chunking → Embeddings → FAISS | Carga los documentos, los fragmenta, genera embeddings y construye el índice vectorial. |
| **B. Consulta (runtime)** | Memoria → Reformulación → Retrieve → Augment → Generate → Post-proceso | Recibe la consulta, recupera información relevante y genera una respuesta basada en el contexto documental. |
| **C. Evaluación** | Métricas + conjunto dorado | Evalúa la calidad de la recuperación y de la generación. |

### Fuentes de información

| Tipo | Ubicación | Descripción |
|---|---|---|
| **Internas** | `data/interno/` | Documentos institucionales y normativa académica. |
| **Externas** | `data/externo/` | Documentos normativos y fuentes públicas relacionadas con beneficios estudiantiles. |

Los documentos internos y externos comparten el mismo pipeline de procesamiento e
índice vectorial, permitiendo recuperar información desde distintas fuentes y
mostrar el origen de los documentos utilizados en cada respuesta.

### Flujo general

```text
                         DOCUMENTOS
                              │
                              ▼
                       ┌─────────────┐
                       │   Ingesta   │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Chunking  │
                       └──────┬──────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │       Ollama       │
                    │  nomic-embed-text  │
                    └──────────┬─────────┘
                               │
                               ▼
                       ┌─────────────┐
                       │    FAISS    │
                       └──────┬──────┘
                              │
                              │
                       CONSULTA USUARIO
                              │
                              ▼
                       ┌─────────────┐
                       │   Memoria   │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │Reformulación│
                       └──────┬──────┘
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
                    ┌──────────────────┐
                    │      Ollama      │
                    │    Gemma 3 4B    │
                    └─────────┬────────┘
                              │
                              ▼
                         RESPUESTA
                    + FUENTES + CONFIANZA

3. Estructura del repositorio
mesa-ayuda-rag/
│
├── config/
│   └── settings.py              # Configuración central del sistema
│
├── data/
│   ├── interno/                 # Corpus institucional
│   └── externo/                 # Corpus normativo externo
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
├── .gitignore
└── README.md

4. Tecnologías utilizadas
Lenguaje
Python 3.10 o superior

Inteligencia Artificial
Ollama — ejecución local de modelos de inteligencia artificial.

Gemma 3 4B — generación de respuestas.

nomic-embed-text — generación de embeddings.

RAG y búsqueda vectorial
LangChain — integración de los componentes del pipeline RAG.

FAISS — almacenamiento y búsqueda vectorial.

Embeddings locales — representación semántica de los documentos.

Procesamiento documental
pypdf

Python Standard Library

Evaluación
Context Precision

Context Recall

Faithfulness

Answer Relevancy

5. Requisitos previos
Antes de ejecutar el proyecto se necesita:

Python 3.10 o superior.

Ollama instalado.

Modelo gemma3:4b.

Modelo nomic-embed-text.

Dependencias Python del proyecto.

Verificar Ollama
Para comprobar que Ollama está instalado:

ollama --version

Descargar los modelos
ollama pull gemma3:4b
ollama pull nomic-embed-text

En Windows, si ollama no está disponible directamente en el PATH, se puede
utilizar su ejecutable instalado localmente:

& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull gemma3:4b
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull nomic-embed-text

Para comprobar que el modelo de generación funciona:

ollama run gemma3:4b

6. Instalación y ejecución
Paso 1 — Clonar el repositorio
git clone https://github.com/cristian250507/Mesa-ayuda-rag.git
cd Mesa-ayuda-rag

Paso 2 — Crear el entorno virtual
python -m venv .venv

Paso 3 — Activar el entorno virtual
Windows PowerShell

.venv\Scripts\Activate.ps1

Linux / macOS

source .venv/bin/activate

Paso 4 — Instalar las dependencias
pip install -r requirements.txt

Paso 5 — Construir el índice vectorial
La primera vez que se ejecuta el proyecto es necesario procesar los documentos:

python -m src.main --indexar

Este proceso:

Carga los documentos desde data/.

Divide los documentos en chunks.

Genera embeddings mediante nomic-embed-text.

Construye el índice FAISS.

Guarda el índice en indice_faiss/.

La indexación debe repetirse cuando se agreguen, eliminen o modifiquen documentos
del corpus.

Paso 6 — Iniciar la atención conversacional
python -m src.main --chat

El sistema permite realizar consultas directamente desde la consola.

Comandos disponibles:

reiniciar

Reinicia el contexto de la conversación.

salir

Finaliza la aplicación.

Paso 7 — Evaluar el sistema
python -m src.main --evaluar

Este comando ejecuta los casos definidos en:

evaluacion/conjunto_dorado.json

y calcula métricas relacionadas con:

Context Precision

Context Recall

Faithfulness

Answer Relevancy

7. Configuración
El proyecto utiliza un archivo .env para definir la configuración de Ollama.

Crear un archivo .env en la raíz del proyecto:

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

También se incluye .env.example como referencia.

Importante: el archivo .env está destinado a la configuración local y
no debe subirse al repositorio.

No se requieren claves de API de OpenAI ni de GitHub para ejecutar el sistema.

8. Ejemplo de uso
Primera consulta
Estudiante > Reprobé dos veces Cálculo, ¿qué me pasa?

El sistema recupera los documentos relevantes y genera una respuesta utilizando
el contexto documental:

Asistente > Reprobar por segunda vez consecutiva una misma asignatura
constituye una causal de eliminación académica según el Reglamento Académico.

Fuentes: Reglamento_Academico (interno)

Confianza: alto | Chunks usados: 3

El estudiante puede realizar una pregunta relacionada:

Estudiante > ¿Y eso afecta mi beca?

El sistema utiliza el contexto de la conversación para interpretar que "eso"
hace referencia a la situación académica mencionada anteriormente.

La consulta se reformula internamente antes de realizar la búsqueda vectorial,
permitiendo recuperar información relacionada con beneficios estudiantiles.

Asistente > Según la normativa recuperada, la situación puede afectar la
mantención del beneficio. Se recomienda revisar las condiciones específicas
del beneficio y, si corresponde, consultar con la Dirección de Asuntos
Estudiantiles.

Fuentes: Normativa_Beneficios_Estudiantiles (externo)

Confianza: alto | Chunks usados: 2

Este ejemplo demuestra la gestión de contexto conversacional, ya que una
pregunta dependiente del contexto anterior puede ser reformulada antes de
realizar la recuperación de información.

9. Parámetros configurables y componentes principales
Los principales parámetros se encuentran centralizados en:

config/settings.py

Parámetro	Valor	Justificación
chunk_size	800	Equilibrio entre conservar contexto y evitar chunks demasiado grandes.
chunk_overlap	120	Permite mantener continuidad entre fragmentos.
top_k	4	Recupera los documentos más relevantes sin saturar el contexto.
umbral_distancia	1.15	Filtra resultados con baja similitud antes de generar una respuesta.
temperature	0.1	Favorece respuestas más deterministas para información normativa.
ventana_turnos	5	Mantiene los últimos turnos de la conversación.
umbral_resumen	10	Define cuándo se intenta generar un resumen de la conversación.

Componentes principales
Ingesta y Chunking
src/ingesta.py

Carga los documentos del corpus y los divide en fragmentos de tamaño controlado
para facilitar la búsqueda semántica.

Embeddings y FAISS
src/vector_store.py

Utiliza nomic-embed-text mediante Ollama para transformar los chunks en vectores
y FAISS para realizar la recuperación de información relevante.

Memoria
src/memoria.py

Mantiene el contexto de la conversación para permitir preguntas de seguimiento y
reformulación de consultas.

Pipeline RAG
src/rag_pipeline.py

Coordina:

Memoria conversacional.

Reformulación de la consulta.

Recuperación de documentos.

Construcción del contexto.

Generación mediante Gemma 3 4B.

Post-procesamiento.

Registro de fuentes.

Nivel de confianza.

Evaluación
src/evaluacion.py

Permite medir por separado la calidad de la recuperación y la calidad de las
respuestas generadas.

Flujo de una consulta
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

Este flujo permite separar claramente la recuperación de información de la
generación de lenguaje, reduciendo el riesgo de que el modelo responda sin
respaldo documental.

10. Ventajas, limitaciones y seguridad
Ventajas de la solución
La arquitectura propuesta permite:

Ejecutar el sistema de forma local.

No depender de una API externa de pago para la generación ni los embeddings.

Mantener control sobre los documentos utilizados como fuente.

Incorporar trazabilidad mediante las fuentes recuperadas.

Mantener contexto conversacional.

Separar recuperación y generación.

Utilizar modelos diferentes para generación y embeddings.

Evaluar el comportamiento del sistema mediante un conjunto de pruebas.

Mantener una arquitectura modular que permite reemplazar componentes en el futuro.

Limitaciones conocidas
El sistema no reemplaza la atención de un funcionario o ejecutivo.

No resuelve casos particulares que requieran interpretación administrativa o
jurídica.

La calidad de las respuestas depende directamente de la calidad y vigencia de
los documentos almacenados en data/.

Un documento desactualizado puede producir una respuesta correcta respecto al
corpus, pero incorrecta respecto a la normativa vigente.

El modelo se ejecuta localmente, por lo que el tiempo de respuesta depende del
hardware disponible.

Las métricas de generación basadas en evaluación automática pueden presentar
variabilidad y deben interpretarse como indicadores de calidad, no como una
medición absoluta.

Seguridad y configuración
El sistema utiliza modelos locales mediante Ollama y no requiere claves de
API externas.

La comunicación con Ollama se realiza mediante el servidor local:

http://localhost:11434

Los modelos utilizados son:

Función	Modelo
Generación	gemma3:4b
Embeddings	nomic-embed-text

El archivo .env contiene únicamente parámetros de configuración local y debe
mantenerse fuera del repositorio.

👥 Autores
Proyecto desarrollado por:

Cristian Pizarro

Marco Avello

Benjamin Fredes

Para la Evaluación Parcial 1 de:

ISY0101 — Ingeniería de Soluciones con IA