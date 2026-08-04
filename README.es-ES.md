# COMA: Cuando la Compresión se Convierte en una Superficie de Ataque

Artifact para el artículo **"Cuando la Compresión se Convierte en una Superficie de Ataque: Ataques de Caja Negra en Agentes de Lenguaje Grande con Prompts Comprimidos (LLM)"**.

## Resumen

COMA es un marco de trabajo adversarial de caja negra que explota la compresión de indicaciones como una superficie de ataque en tuberías de agentes de lenguaje grande (LLM). Demuestra que un adversario puede crear entradas que, después de la compresión, eliminan selectivamente o corrompen información crítica, causando que los agentes LLM downstream tengan un comportamiento inadecuado.

El marco de trabajo implementa un ataque de dos etapas:
- **Etapa I (Selección de Objetivos):** Identificar qué información suprimir (segmentos de respuesta, negaciones de reglas de seguridad, palabras clave críticas de preferencias)
- **Etapa II (Búsqueda de Preimagenes):** Usar optimización basada en COMA para encontrar entradas adversarias que, después de la compresión, coincidan con el objetivo

### Tareas
| Tarea | Abreviatura | Descripción |
|------|---------|-------------|
| Selección de Herramientas del Agente | ATS | Manipular qué herramienta/producto recomienda el agente |
| Respuesta a Preguntas | QA | Suprimir segmentos de respuesta para que el agente no pueda responder correctamente |
| Corrupción de Indicaciones del Sistema | SPC | Eliminar negaciones de reglas de seguridad (ej. "no" -> "") para deshabilitar reglas de seguridad |

### Compresores Evaluados
| Tipo | Compresor | Modelo Sustituto |
|------|-----------|-----------------|
| Extractivo | LLMLingua-1 | Llama-2-7B |
| Extractivo | LLMLingua-2 | xlm-roberta-large |
| Extractivo | SelectiveContext | Llama-2-7B |
| Abstractivo | Qwen3-4B | Qwen3-4B |
| Abstractivo | Llama-3.2-3B | Llama-3.2-3B-Instruct |
| Abstractivo | Gemma-3-4B | Gemma-3-4B |

## Inicio Rápido

### 1. Instalación

```bash
conda create -n coma python=3.10 -y && conda activate coma
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd artifact/
pip install -e ".[all]"
```

## Reproduciendo los Resultados del Artículo

Cada RQ tiene un script de reproducción dedicado:

| RQ | Script | Descripción |
|----|--------|-------------|
| RQ1 | `scripts/reproduce_rq1.sh` | Efectividad: 3 tareas x 6 compresores |
| RQ2 | `scripts/reproduce_rq2.sh` | Generalización: barrido de presupuestos + LLMs backend |
| RQ3 | `scripts/reproduce_rq3.sh` | Incongruencia de modelos sustitutos + retención de tokens |
| RQ4 | `scripts/reproduce_rq4.sh` | Estudios de caso: VSCode Cline, LangChain+Ollama |
| RQ5 | `scripts/reproduce_rq5.sh` | Evaluación de defensas |

## Formato de Salida

Los resultados de los ataques se guardan como archivos JSONL, una entrada por línea:

```json
{
    "context": "texto de entrada original...",
    "attacked_context": "texto de entrada adversarial...",
    "best_loss": 0.023,
    "converged": true,
    "steps": 142
}
```

## Estructura de Archivos

```
artifact/
  README.md                 
  config.py                 # Configuración (basada en variables de entorno, sin rutas codificadas)
  requirements.txt          # Dependencias de Python
  .gitignore

  comattack/                # Paquete principal de Python
    __init__.py
    attacks/                # Etapa II: búsqueda de preimagenes (extractivo + abstractivo)
    compressors/            # Envolturas de compresores
    defense/                # Líneas base de defensa
    evaluation/             # Evaluación de extremo a extremo, métricas, cumplimiento
    llm/                    # Abstracción de proveedores de LLM (vLLM, OpenAI, Ollama)
    targets/                # Etapa I: generación de objetivos por tarea
    data/                   # Datos del paquete (prompts, plantillas)

  run_guardrail_attack.py   # Punto de entrada: tarea SPC
  run_qa_attack.py          # Punto de entrada: tarea QA
  run_pref_attack.py        # Punto de entrada: tarea ATS
  run_surrogate_mismatch.py # Punto de entrada: incompatibilidad de modelos sustitutos
  run_pref_attack.sh        # Lanzador por lotes: ATS (todos los compresores)
  run_qa_attack.sh          # Lanzador por lotes: QA (todos los compresores)
  run_surrogate_mismatch.sh  # Lanzador por lotes: cuadrícula de modelos sustitutos
```
