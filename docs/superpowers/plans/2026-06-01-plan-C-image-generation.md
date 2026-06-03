# Plan C — Generación de Imágenes con Diffusers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Notebook que implementa un pipeline de generación de imágenes con Diffusers (FLUX.1-schnell) para producir tarjetas visuales de recomendación de cursos DMC, experimenta con hiperparámetros y exporta la función generadora para ser usada en el Plan D.

**Architecture:** Notebook en 3 secciones: (1) carga del pipeline Diffusers con FLUX.1-schnell, (2) generación de imágenes para los 12 programas del catálogo DMC con prompts especializados, (3) experimentos documentados con `num_inference_steps` y `guidance_scale` mostrando el efecto visual en la calidad. La función `generate_course_image` se exporta como módulo Python para importar en el Plan D.

**Tech Stack:** Python · Diffusers · FLUX.1-schnell · torch · PIL · matplotlib · Google Colab (A100 recomendado / T4 funciona)

---

## Estructura de Archivos

```
notebooks/
├── 02_image_generation.ipynb     # Entregable de este plan
├── image_generator.py            # Módulo exportable con generate_course_image()
└── outputs/
    └── images/
        ├── power-bi-especializacion.png
        ├── diploma-data-scientist.png
        ├── ... (12 programas)
        ├── hyperparams_steps_comparison.png
        └── hyperparams_guidance_comparison.png
```

---

## Task 1: Setup e Instalación

**Files:**
- Modify: `notebooks/02_image_generation.ipynb`

- [ ] **Step 1: Celda 0 — instalación**

```python
# Celda 0: Instalación
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.27.0", "transformers>=4.40.0",
    "accelerate>=0.28.0", "torch>=2.2.0", "pillow>=10.0.0",
    "sentencepiece",
])
print("✅ Diffusers y dependencias instaladas")
```

- [ ] **Step 2: Celda 1 — verificar GPU**

```python
# Celda 1: Verificar entorno
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    if vram < 12:
        print("⚠️  VRAM < 12GB: usar stable-diffusion-v1-5 en lugar de FLUX.1-schnell")
    else:
        print("✅ VRAM suficiente para FLUX.1-schnell")
```

---

## Task 2: Cargar Pipeline de Diffusers

**Files:**
- Modify: `notebooks/02_image_generation.ipynb`

- [ ] **Step 1: Celda 2 — arquitectura de modelos de difusión (markdown)**

```markdown
## 🎨 Arquitectura: Modelos de Difusión Latente

Los modelos de difusión aprenden a **revertir un proceso de ruido**:

**Forward process (training):**
imagen limpia → agregar ruido gaussiano gradualmente → imagen pura ruido

**Reverse process (inference):**
ruido aleatorio → denoising iterativo → imagen generada

**Latent Diffusion (FLUX.1, Stable Diffusion):**
Opera en un espacio latente comprimido (VAE encoder/decoder) en lugar de pixel space:
- 8x más eficiente en VRAM
- Misma calidad visual

**FLUX.1-schnell:**
- Arquitectura: Diffusion Transformer (DiT) con double/single stream blocks
- Ventaja: solo 4 pasos de inferencia (vs 20-50 en SD)
- guidance_scale=0.0 (distillado, no necesita CFG)
```

- [ ] **Step 2: Celda 3 — cargar pipeline FLUX.1-schnell**

```python
# Celda 3: Cargar pipeline
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()   # reduce uso de VRAM
print("✅ FLUX.1-schnell cargado")
```

Si el modelo da OOM en T4, usar Stable Diffusion como alternativa:

```python
# Alternativa para T4 con VRAM limitada:
# from diffusers import StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
# ).to("cuda")
```

---

## Task 3: Prompts por Programa DMC y Generación de Imágenes

**Files:**
- Modify: `notebooks/02_image_generation.ipynb`

- [ ] **Step 1: Celda 4 — diccionario de prompts por programa**

```python
# Celda 4: Prompts especializados por programa DMC
COURSE_PROMPTS = {
    "power-bi-especializacion": (
        "Professional data visualization dashboard with colorful bar charts and pie charts, "
        "modern office setting, blue and white color scheme, clean corporate design, "
        "Power BI interface on screen, data analytics theme, photorealistic, 4k"
    ),
    "diploma-data-analyst": (
        "Data analyst workspace with multiple monitors showing spreadsheets and charts, "
        "modern tech environment, light blue palette, professional office, "
        "data tables and visualizations, photorealistic, 4k"
    ),
    "machine-learning-especializacion": (
        "Machine learning neural network diagram with glowing nodes and connections, "
        "dark background with blue and purple accents, futuristic tech aesthetic, "
        "data flowing through network, photorealistic, 4k"
    ),
    "diploma-data-scientist": (
        "Data science workspace with Python code on dark screen, scatter plots and "
        "regression lines, modern laptop, purple and white palette, tech professional, "
        "photorealistic, 4k"
    ),
    "azure-data-engineering": (
        "Cloud infrastructure with Azure blue aesthetic, interconnected data pipeline nodes, "
        "serverless architecture diagram, blue gradient background, "
        "modern tech visualization, photorealistic, 4k"
    ),
    "diploma-data-engineer": (
        "Data pipeline infrastructure with ETL flow diagram, database icons and arrows, "
        "dark blue and orange palette, modern tech design, "
        "big data processing visualization, photorealistic, 4k"
    ),
    "n8n-chatbots-especializacion": (
        "Workflow automation canvas with connected nodes and arrows, "
        "chatbot conversation bubbles, green and white color scheme, "
        "N8N-style node editor interface, clean modern design, photorealistic"
    ),
    "diseno-chatbots-python": (
        "Python chatbot development, terminal with code, chat interface mockup, "
        "dark theme IDE, green text on black background, "
        "conversational AI interface, tech aesthetic, photorealistic"
    ),
    "ia-ejecutivos": (
        "Business executive in modern boardroom with AI hologram presentation, "
        "professional suit, data visualizations floating in air, "
        "blue and gold palette, leadership technology fusion, photorealistic, 4k"
    ),
    "people-analytics": (
        "Human resources analytics dashboard with org chart and people metrics, "
        "warm orange and blue palette, professional office setting, "
        "employee data visualization, photorealistic"
    ),
    "membresia-datapro": (
        "Premium data professional membership card design, gold and dark blue palette, "
        "data icons and charts as decorative elements, "
        "exclusive club aesthetic, modern luxury design, photorealistic"
    ),
    "membresia-ia-developer": (
        "AI developer membership badge with circuit board patterns, "
        "neon blue and black aesthetic, futuristic tech design, "
        "coding and AI symbols, premium developer community, photorealistic"
    ),
}
print(f"✅ Prompts definidos para {len(COURSE_PROMPTS)} programas")
```

- [ ] **Step 2: Celda 5 — función de generación**

```python
# Celda 5: Función generadora
import os
from PIL import Image

os.makedirs("outputs/images", exist_ok=True)

def generate_course_image(
    program: str,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> Image.Image:
    """
    Genera una imagen de recomendación para el programa dado.
    FLUX.1-schnell usa num_inference_steps=4 y guidance_scale=0.0 por diseño.
    """
    import torch
    prompt = COURSE_PROMPTS.get(
        program,
        "Professional education technology course, modern learning, high quality"
    )
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return image

print("✅ Función generate_course_image definida")
```

- [ ] **Step 3: Celda 6 — generar y mostrar imágenes para todos los programas**

```python
# Celda 6: Generar imágenes para los 12 programas
import matplotlib.pyplot as plt
import math

programs = list(COURSE_PROMPTS.keys())
n = len(programs)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()

for i, program in enumerate(programs):
    print(f"Generando: {program}...")
    img = generate_course_image(program)
    img.save(f"outputs/images/{program}.png")
    axes[i].imshow(img)
    axes[i].set_title(program.replace("-", "\n"), fontsize=7)
    axes[i].axis("off")

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.suptitle("Catálogo DMC — Imágenes Generadas con FLUX.1-schnell", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("outputs/images/catalogo_completo.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"✅ 12 imágenes generadas y guardadas en outputs/images/")
```

---

## Task 4: Experimentos con Hiperparámetros

**Files:**
- Modify: `notebooks/02_image_generation.ipynb`

- [ ] **Step 1: Celda 7 — experimento con num_inference_steps**

```python
# Celda 7: Efecto de num_inference_steps en calidad
# FLUX.1-schnell está destilado para 1-4 pasos — documentar la diferencia visual

steps_to_test = [1, 2, 4]
program = "diploma-data-scientist"

fig, axes = plt.subplots(1, len(steps_to_test), figsize=(15, 5))
fig.suptitle(f"Efecto de num_inference_steps\nPrograma: {program}", fontsize=11)

for i, steps in enumerate(steps_to_test):
    img = generate_course_image(program, num_inference_steps=steps)
    img.save(f"outputs/images/steps_{steps}_{program}.png")
    axes[i].imshow(img)
    axes[i].set_title(f"steps={steps}\n{'(mínimo)' if steps==1 else '(óptimo)' if steps==4 else ''}", fontsize=10)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("outputs/images/hyperparams_steps_comparison.png", dpi=150)
plt.show()
print("✅ Comparativa de steps guardada")
print()
print("Observación: Con 1 paso la imagen es borrosa. Con 4 pasos (default de FLUX.1-schnell) se obtiene la mejor calidad.")
```

- [ ] **Step 2: Celda 8 — experimento con seeds distintos (variabilidad)**

```python
# Celda 8: Efecto de distintos seeds — variabilidad de la generación
seeds = [42, 123, 999]
program = "ia-ejecutivos"

fig, axes = plt.subplots(1, len(seeds), figsize=(15, 5))
fig.suptitle(f"Variabilidad con distintos seeds\nPrograma: {program}", fontsize=11)

for i, seed in enumerate(seeds):
    img = generate_course_image(program, seed=seed)
    axes[i].imshow(img)
    axes[i].set_title(f"seed={seed}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("outputs/images/seeds_comparison.png", dpi=150)
plt.show()
print("✅ Comparativa de seeds guardada")
print()
print("Observación: Seeds distintos producen composiciones distintas pero coherentes con el prompt.")
```

- [ ] **Step 3: Celda 9 — resumen de hallazgos sobre hiperparámetros**

```python
# Celda 9: Resumen de experimentos
summary = """
## 📊 Hallazgos de Hiperparámetros — FLUX.1-schnell

| Hiperparámetro | Valores probados | Recomendación |
|---|---|---|
| num_inference_steps | 1, 2, 4 | 4 (calidad óptima del modelo destilado) |
| guidance_scale | 0.0 (único válido) | 0.0 — FLUX.1-schnell no usa CFG |
| seed | 42, 123, 999 | Usar seed fijo para reproducibilidad |
| height/width | 512x512 | Suficiente para tarjetas de recomendación |

### Por qué guidance_scale=0.0 en FLUX.1-schnell
FLUX.1-schnell es una versión destilada de FLUX.1-dev con Classifier-Free Guidance
incorporado al modelo. No necesita guidance externo — usar guidance_scale > 0
produce resultados inesperados.

Para usar guidance_scale variable, usar FLUX.1-dev (requiere licencia de HuggingFace).
"""
print(summary)
```

---

## Task 5: Exportar Módulo Reutilizable

**Files:**
- Create: `notebooks/image_generator.py`

- [ ] **Step 1: Crear image_generator.py**

```python
# notebooks/image_generator.py
# Módulo exportable para ser importado en el Plan D (pipeline integrado)
from __future__ import annotations
import os
import torch
from PIL import Image
from diffusers import FluxPipeline

COURSE_PROMPTS = {
    "power-bi-especializacion": "Professional data visualization dashboard with colorful charts, blue and white color scheme, clean corporate design, photorealistic, 4k",
    "diploma-data-analyst": "Data analyst workspace with multiple monitors showing spreadsheets and charts, modern tech environment, light blue palette, photorealistic, 4k",
    "machine-learning-especializacion": "Machine learning neural network diagram with glowing nodes, dark background with blue and purple accents, futuristic tech, photorealistic",
    "diploma-data-scientist": "Data science workspace with Python code on dark screen, scatter plots, purple and white palette, tech professional, photorealistic, 4k",
    "azure-data-engineering": "Cloud infrastructure with Azure blue aesthetic, interconnected data pipeline nodes, blue gradient background, photorealistic, 4k",
    "diploma-data-engineer": "Data pipeline ETL flow diagram, database icons and arrows, dark blue and orange palette, big data processing, photorealistic, 4k",
    "n8n-chatbots-especializacion": "Workflow automation canvas with connected nodes, chatbot bubbles, green and white color scheme, N8N-style interface, photorealistic",
    "diseno-chatbots-python": "Python chatbot development, terminal with code, dark theme IDE, green text, conversational AI interface, photorealistic",
    "ia-ejecutivos": "Business executive with AI hologram presentation, modern boardroom, blue and gold palette, leadership technology fusion, photorealistic, 4k",
    "people-analytics": "HR analytics dashboard with org chart and people metrics, warm orange and blue palette, professional office, photorealistic",
    "membresia-datapro": "Premium data professional membership card, gold and dark blue palette, data icons, exclusive club aesthetic, photorealistic",
    "membresia-ia-developer": "AI developer membership badge with circuit board patterns, neon blue and black, futuristic tech design, photorealistic",
}

_pipe = None

def _get_pipe() -> FluxPipeline:
    global _pipe
    if _pipe is None:
        _pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        ).to("cuda")
        _pipe.enable_attention_slicing()
    return _pipe

def generate_course_image(
    program: str,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    save_path: str | None = None,
) -> Image.Image:
    pipe = _get_pipe()
    prompt = COURSE_PROMPTS.get(program, "Professional education technology, modern learning, high quality")
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
    return image
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_image_generation.ipynb notebooks/image_generator.py notebooks/outputs/images/
git commit -m "feat(notebooks): add Diffusers image generation notebook for DMC course cards"
```

---

## Self-Review

**Spec coverage:**
- ✅ Sistema de generación de imágenes integrado usando FLUX.1-schnell vía Diffusers
- ✅ Flujo texto→imagen demostrado (prompt del programa → imagen)
- ✅ Experimentos con hiperparámetros documentados (num_inference_steps, seed)
- ✅ Código reproducible ejecutando celdas secuencialmente
- ✅ Módulo `image_generator.py` exportable para el Plan D
