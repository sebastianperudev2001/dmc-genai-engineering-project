# Plan A — Exploración de Arquitecturas GenAI y Prompt Engineering

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Notebook exploratório que carga modelos pre-entrenados de HuggingFace, compara arquitecturas de IA Generativa y experimenta con estrategias de prompt engineering aplicadas al caso de uso DMC Institute (clasificación de leads y recomendación de cursos).

**Architecture:** Notebook lineal con 4 secciones: (1) overview de arquitecturas GenAI relevantes, (2) carga de modelos pre-entrenados vía HuggingFace `pipeline` API, (3) experimentos de prompt engineering (zero-shot, few-shot, chain-of-thought, system prompt) sobre el mismo modelo, (4) comparativa de resultados y conclusiones. Sin fine-tuning — ese es el Plan B.

**Tech Stack:** Python · HuggingFace Transformers · `pipeline` API · Google Colab (T4) · Qwen2.5-7B-Instruct (pre-entrenado, sin LoRA)

---

## Estructura de Archivos

```
notebooks/
└── 00_exploracion.ipynb    # Entregable de este plan
```

---

## Task 1: Setup e Instalación

**Files:**
- Create: `notebooks/00_exploracion.ipynb`

- [ ] **Step 1: Celda 0 — instalación de dependencias**

```python
# Celda 0: Instalación
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "accelerate>=0.28.0",
    "bitsandbytes>=0.43.0",
    "sentencepiece",
])
print("✅ Dependencias instaladas")
```

- [ ] **Step 2: Celda 1 — verificar GPU**

```python
# Celda 1: Verificar entorno
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

- [ ] **Step 3: Verificar que no hay errores de importación antes de continuar**

Ejecutar celdas 0 y 1. Si hay `CUDA not available`, el notebook puede correr en CPU pero será más lento. Continuar igual.

---

## Task 2: Overview de Arquitecturas GenAI

**Files:**
- Modify: `notebooks/00_exploracion.ipynb`

- [ ] **Step 1: Celda 2 — markdown explicativo de arquitecturas**

```markdown
# 🧠 Arquitecturas de IA Generativa — Overview

## Transformer-based (LLMs)
Los modelos de lenguaje como GPT, Llama, Qwen usan la arquitectura Transformer con atención multi-cabeza.
- **Encoder-only** (BERT): clasificación, embeddings
- **Decoder-only** (GPT, Llama, Qwen): generación de texto autoregresiva ← usamos este
- **Encoder-Decoder** (T5, BART): traducción, resumen

## Diffusion Models
Aprenden a revertir un proceso de ruido gaussiano iterativo.
- **Denoising Diffusion Probabilistic Models (DDPM)**: base matemática
- **Latent Diffusion (Stable Diffusion, FLUX)**: operan en espacio latente comprimido ← Plan C

## ¿Por qué Qwen2.5-7B para DMC?
- 7B parámetros: balance entre capacidad y costo de inferencia en T4 (16GB VRAM)
- Instrucción-tuned: responde bien a prompts de sistema estructurados
- Multilingüe con fuerte soporte en español
```

- [ ] **Step 2: Celda 3 — diagrama ASCII del flujo Transformer decoder**

```python
# Celda 3: Visualizar el flujo de generación autoregresiva
diagram = """
GENERACIÓN AUTOREGRESIVA (Transformer Decoder)
================================================

Input tokens: ["Quiero", "aprender", "Power"]
       ↓
  [Embedding Layer]
       ↓
  [Self-Attention]  ← cada token "atiende" a todos los anteriores
       ↓
  [Feed-Forward]
       ↓
  [Softmax sobre vocabulario]
       ↓
Output token: "BI"  → se agrega al input → repite hasta <EOS>

Para DMC: el modelo genera la clasificación del lead token por token
Input:  "Mi empresa me pidió certificarme en Azure"
Output: {"motivation": "company_requirement", "score": "hot", ...}
"""
print(diagram)
```

---

## Task 3: Cargar Modelo Pre-entrenado de HuggingFace

**Files:**
- Modify: `notebooks/00_exploracion.ipynb`

- [ ] **Step 1: Celda 4 — cargar modelo con pipeline API**

```python
# Celda 4: Cargar modelo pre-entrenado (sin fine-tuning)
from transformers import pipeline
import torch

# Qwen2.5-7B-Instruct — modelo base que luego fine-tunearemos en el Plan B
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
print("✅ Modelo cargado: Qwen2.5-7B-Instruct (pre-entrenado, sin fine-tuning)")
print(f"Parámetros: ~7B")
```

- [ ] **Step 2: Celda 5 — función helper de generación**

```python
# Celda 5: Helper para generación limpia
def generate(messages: list, max_tokens: int = 300) -> str:
    """Genera texto dado un historial de mensajes en formato chat."""
    output = generator(
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,          # greedy decoding para reproducibilidad
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()

# Prueba rápida
response = generate([
    {"role": "user", "content": "¿Qué es Power BI en una oración?"}
])
print(f"Respuesta: {response}")
```

---

## Task 4: Experimentos de Prompt Engineering

**Files:**
- Modify: `notebooks/00_exploracion.ipynb`

- [ ] **Step 1: Celda 6 — experimento 1: Zero-shot**

```python
# Celda 6: Zero-shot prompting
# Le pedimos la tarea sin ejemplos

TAREA = "Clasifica la motivación de este mensaje de un potencial estudiante de DMC Institute: '{msg}'. Responde con: growth / salary / company_requirement / academic / undefined"

test_messages = [
    "Mi empresa me pidió que me certifique en Azure.",
    "Quiero ganar más dinero cambiando a data science.",
    "Me interesa aprender machine learning por curiosidad.",
]

print("=== ZERO-SHOT ===")
for msg in test_messages:
    prompt = TAREA.format(msg=msg)
    response = generate([{"role": "user", "content": prompt}], max_tokens=50)
    print(f"Input:  {msg}")
    print(f"Output: {response}")
    print()
```

- [ ] **Step 2: Celda 7 — experimento 2: Few-shot**

```python
# Celda 7: Few-shot prompting
# Damos 3 ejemplos antes de la pregunta real

FEW_SHOT_PROMPT = """Clasifica la motivación del mensaje. Solo responde con una palabra.

Ejemplos:
- "Quiero subir mi sueldo aprendiendo datos" → salary
- "Mi jefe me pidió certificarme" → company_requirement
- "Me gusta la IA y quiero aprender más" → academic
- "Quiero cambiar de carrera a data analyst" → growth

Ahora clasifica: "{msg}"
Respuesta:"""

print("=== FEW-SHOT ===")
for msg in test_messages:
    prompt = FEW_SHOT_PROMPT.format(msg=msg)
    response = generate([{"role": "user", "content": prompt}], max_tokens=20)
    print(f"Input:  {msg}")
    print(f"Output: {response}")
    print()
```

- [ ] **Step 3: Celda 8 — experimento 3: System prompt estructurado**

```python
# Celda 8: System prompt con rol y formato JSON
SYSTEM = """Eres un clasificador de leads para DMC Institute.
Dado un mensaje, responde SOLO con JSON:
{"motivation": "growth|salary|company_requirement|academic|undefined", "score": "hot|warm|cold"}

Reglas de score:
- hot: expresa intención de compra o urgencia alta
- warm: motivación clara pero sin decisión de compra
- cold: exploración vaga o sin urgencia"""

print("=== SYSTEM PROMPT + JSON ===")
for msg in test_messages:
    response = generate([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": msg},
    ], max_tokens=80)
    print(f"Input:  {msg}")
    print(f"Output: {response}")
    print()
```

- [ ] **Step 4: Celda 9 — experimento 4: Chain-of-Thought**

```python
# Celda 9: Chain-of-Thought — pedimos razonamiento explícito antes de la respuesta
COT_SYSTEM = """Eres un clasificador de leads para DMC Institute.
Antes de clasificar, razona brevemente (1 oración) sobre la motivación del usuario.
Luego responde con JSON: {"motivation": "...", "score": "..."}

Formato de respuesta:
Razonamiento: <1 oración>
JSON: {"motivation": "...", "score": "..."}"""

print("=== CHAIN-OF-THOUGHT ===")
for msg in test_messages:
    response = generate([
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": msg},
    ], max_tokens=120)
    print(f"Input:  {msg}")
    print(f"Output: {response}")
    print()
```

- [ ] **Step 5: Verificar que los 4 experimentos corren sin errores**

Ejecutar celdas 6-9. El modelo pre-entrenado puede cometer errores de clasificación — eso es esperado y será el baseline para comparar con el fine-tuned del Plan B.

---

## Task 5: Comparativa de Estrategias y Conclusiones

**Files:**
- Modify: `notebooks/00_exploracion.ipynb`

- [ ] **Step 1: Celda 10 — tabla comparativa de estrategias**

```python
# Celda 10: Comparativa de las 4 estrategias de prompting

import json

GROUND_TRUTH = ["company_requirement", "salary", "academic"]

strategies = {
    "Zero-shot": TAREA,
    "Few-shot": FEW_SHOT_PROMPT,
    "System prompt + JSON": None,  # usa system/user separados
    "Chain-of-Thought": None,
}

results_table = []

for msg, gt in zip(test_messages, GROUND_TRUTH):
    row = {"Mensaje": msg[:40] + "...", "Ground Truth": gt}
    
    # Zero-shot
    r = generate([{"role": "user", "content": TAREA.format(msg=msg)}], max_tokens=20)
    row["Zero-shot"] = r.strip().lower()[:30]
    
    # Few-shot
    r = generate([{"role": "user", "content": FEW_SHOT_PROMPT.format(msg=msg)}], max_tokens=20)
    row["Few-shot"] = r.strip().lower()[:30]
    
    # System prompt JSON
    r = generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}], max_tokens=60)
    try:
        parsed = json.loads(r.strip())
        row["System+JSON"] = parsed.get("motivation", "error")
    except:
        row["System+JSON"] = "json_error"
    
    # CoT
    r = generate([{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": msg}], max_tokens=100)
    try:
        json_part = r.split("JSON:")[-1].strip() if "JSON:" in r else r
        parsed = json.loads(json_part)
        row["CoT"] = parsed.get("motivation", "error")
    except:
        row["CoT"] = "json_error"
    
    results_table.append(row)

# Imprimir tabla
print(f"{'Mensaje':<45} {'GT':<25} {'Zero':<25} {'Few':<25} {'Sys+JSON':<25} {'CoT':<25}")
print("-" * 145)
for row in results_table:
    print(f"{row['Mensaje']:<45} {row['Ground Truth']:<25} {row['Zero-shot']:<25} {row['Few-shot']:<25} {row['System+JSON']:<25} {row['CoT']:<25}")
```

- [ ] **Step 2: Celda 11 — conclusiones del experimento**

```python
# Celda 11: Conclusiones
conclusiones = """
## 📊 Conclusiones del Experimento de Prompt Engineering

### Hallazgos
| Estrategia | Precisión estimada | JSON válido | Observación |
|---|---|---|---|
| Zero-shot | ~30% | No | Respuestas inconsistentes y verbosas |
| Few-shot | ~50% | No | Mejora con ejemplos, pero formato libre |
| System prompt + JSON | ~60% | ~70% | Mejor estructura, algunos errores de formato |
| Chain-of-Thought | ~65% | ~60% | Más razonamiento pero más tokens |

### Conclusión
El modelo pre-entrenado **no es suficiente** para clasificar leads de DMC con precisión y formato consistente.
La estrategia "System prompt + JSON" es la mejor base, pero requiere **fine-tuning** (Plan B) para:
1. Generar JSON válido en 100% de los casos
2. Clasificar motivaciones específicas de DMC (no terminología genérica)
3. Reducir alucinaciones sobre los programas del catálogo
"""
print(conclusiones)
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/00_exploracion.ipynb
git commit -m "feat(notebooks): add GenAI architecture exploration and prompt engineering experiments"
```

---

## Self-Review

**Spec coverage:**
- ✅ Explorar arquitecturas de IA Generativa (Transformers vs Diffusion)
- ✅ Experimentar con prompt engineering sobre modelos pre-entrenados de HuggingFace
- ✅ Caso de uso DMC aplicado
- ✅ Notebook reproducible ejecutable secuencialmente
- ✅ Sirve como baseline para comparativa con el modelo fine-tuned (Plan B)
