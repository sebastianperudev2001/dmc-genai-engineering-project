# DMC AI Sales Agent — Pipeline Generativo Multimodal

**Programa:** Diploma AI Engineer — DMC Institute  
**Autores:** 
- Sebastian Chavarry
- Daniel Guardia
- Miguel Anay  
**Entorno de ejecución:** Google Colab (T4/A100)

---

## Problema a Resolver

El equipo comercial de DMC Institute atiende manualmente consultas repetitivas sobre qué curso tomar, cuánto cuesta y qué perfil se necesita — principalmente vía WhatsApp y formulario de contacto. El visitante que llega al sitio no recibe orientación inmediata personalizada y cae en un funnel pasivo: formulario → esperar llamada → posible venta.

---

## Descripción Detallada del Problema

- El equipo dedica ~4 horas/día a calificación y orientación de leads
- La cobertura es limitada: 8h/día, 6 días/semana — sin atención nocturna ni fines de semana
- ~15 leads/día atendidos con una tasa de conversión del 12%
- La recomendación de programa varía según el perfil del visitante (motivación, experiencia, urgencia), lo que requiere un proceso personalizado que hoy depende enteramente de personas
- No existe captura estructurada de datos del lead: nombre, motivación, intención de compra y programa de interés se pierden o quedan en conversaciones de WhatsApp no sistematizadas

---

## Justificación de la Elección del Problema

DMC Institute tiene un catálogo de 12+ programas con perfiles de comprador distintos. La recomendación correcta depende de detectar la motivación del visitante (crecimiento profesional, aumento salarial, requerimiento de empresa o actualización académica), su perfil técnico actual y su nivel de urgencia — exactamente el tipo de tarea de clasificación y generación de texto personalizado que los LLMs resuelven bien.

Adicionalmente, el problema tiene alta frecuencia, patrón predecible y alto impacto económico, lo que lo convierte en un caso de uso ideal para demostrar fine-tuning eficiente, prompt engineering y generación de imágenes en un contexto de negocio real.

---

## Propuesta de Solución

Un pipeline generativo multimodal estructurado en 4 notebooks:

| Notebook | Descripción |
|---|---|
| `00_exploracion.ipynb` | Exploración de arquitecturas de IA Generativa y experimentos de prompt engineering sobre modelos pre-entrenados de HuggingFace |
| `01_finetuning.ipynb` | Dataset de 200+ conversaciones de calificación de leads DMC + fine-tuning de Qwen2.5-7B-Instruct con Unsloth/LoRA |
| `02_image_generation.ipynb` | Pipeline de generación de imágenes con FLUX.1-schnell (Diffusers) para tarjetas visuales de recomendación de cursos |

---

## Arquitectura del Sistema

```
Mensaje del usuario
        │
        ▼
Clasificador de Intención
(Qwen2.5-7B + LoRA fine-tuned)
        │
        ├── Motivación detectada (salary / growth / company / academic)
        ├── Score del lead (hot / warm / cold)
        └── Programa recomendado
                │
                ▼
Generador de Imagen
(FLUX.1-schnell via Diffusers)
                │
                ▼
Tarjeta visual del programa recomendado
```

**Flujo de entrenamiento (Plan B):**
```
Dataset DMC (200 ejemplos JSON)
        │
        ▼
Qwen2.5-7B-Instruct (base, 4-bit quantizado)
        │
   [LoRA r=16]
        │
        ▼
Fine-tuning con SFTTrainer (Unsloth)
        │
        ▼
Adapter LoRA exportado
```

---

## Herramientas y Tecnologías

| Componente | Tecnología |
|---|---|
| LLM base | Qwen2.5-7B-Instruct (HuggingFace) |
| Fine-tuning | Unsloth + LoRA (PEFT) + TRL SFTTrainer |
| Quantización | bitsandbytes (4-bit QLoRA) |
| Generación de imágenes | FLUX.1-schnell (Diffusers) |
| Dataset | 200 conversaciones DMC en formato JSON |
| Entorno | Google Colab (T4/A100) |
| Lenguaje | Python 3.11 |
| Librerías core | Transformers · Datasets · PEFT · Accelerate |

---

## Ejecución

### Requisitos
- Cuenta de Google con acceso a Google Colab
- GPU T4 (16GB, mínimo) o A100 (40GB, recomendado)
- Token de HuggingFace (`HF_TOKEN`) con acceso a modelos gated (necesario para `02_image_generation.ipynb`)

### Pasos

**1. Clonar el repositorio y subir a Google Drive (o usar directamente en Colab):**
```bash
git clone <url-del-repo>
```

**2. Abrir cada notebook en Google Colab con GPU habilitada:**
En Colab: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4 o A100)`

---

#### `00_exploracion.ipynb`
- No requiere outputs previos ni archivos adicionales.
- Instala sus dependencias automáticamente en la primera celda.

#### `01_finetuning.ipynb` — Fine-tuning con Unsloth/LoRA
- Instala sus dependencias automáticamente en la primera celda.
- Requiere el dataset `notebooks/dataset/dmc_leads.json` (ya incluido en el repo). Si subes el notebook solo, también sube la carpeta `dataset/` al mismo directorio en Colab.
- **Opcional:** para publicar el adapter en HuggingFace Hub, agrega `HF_TOKEN` en Colab Secrets (`Herramientas → Secrets`).
- Genera en `outputs/`:
  - `lora_adapter/` — adapter LoRA exportado
  - `base_vs_finetuned.png` — gráfica comparativa

#### `02_image_generation.ipynb` — Generación de imágenes con Diffusers
- Instala sus dependencias automáticamente en la primera celda.
- Requiere `HF_TOKEN` con acceso al modelo `ByteDance/SDXL-Lightning`. Configúralo en la celda de login o en Colab Secrets.
- No depende de outputs del notebook anterior.
- Genera en `outputs/images/`:
  - Una imagen `.png` por cada uno de los 12 programas DMC
  - `catalogo_completo.png` — grilla con todos los programas
  - `hyperparams_steps_comparison.png` y `seeds_comparison.png` — experimentos

---

## Conclusiones y Posibles Mejoras Futuras

### Conclusiones

**Fine-tuning (01_finetuning.ipynb):**
- El fine-tuning con Unsloth/LoRA (r=16) sobre 200 ejemplos en una T4 tomó 355s (~6 min) y redujo el loss de ~2.0 a 0.0721 en 3 epochs
- Solo el 0.92% de los parámetros fueron entrenables (40M de 7.6B), lo que confirma la eficiencia de QLoRA para adaptación de dominio
- El modelo base ya respondía JSON válido en 5/5 casos, pero clasificaba incorrectamente motivación y score en 2/5. El modelo fine-tuned corrigió estos errores, logrando clasificación correcta en los 5 casos de prueba
- El mayor beneficio del fine-tuning no fue el formato (el base ya lo cumplía) sino la precisión semántica: distingue correctamente `company_requirement` vs `growth`, y calibra mejor el score `hot/warm/cold`

**Generación de imágenes (02_image_generation.ipynb):**
- SDXL-Lightning (ByteDance, 4 pasos) genera imágenes de 1024×1024 con coherencia visual suficiente para tarjetas de recomendación de cursos
- Con `num_inference_steps=4`, la calidad es notablemente superior a 1-2 pasos sin costo adicional significativo en T4
- Seeds distintos producen composiciones variadas pero coherentes con el prompt, lo que permite seleccionar la mejor variante sin re-prompt

### Mejoras Futuras
- Aumentar el dataset a 1000+ ejemplos con conversaciones multi-turno para mejorar el comportamiento conversacional del agente
- Evaluar el fine-tuned en un conjunto de validación separado (actualmente solo se evalúa en 5 casos de prueba incluidos en el entrenamiento)
- Integrar RAG con los brochures reales de DMC (S3 + embeddings) para fundamentar las respuestas en documentos oficiales
- Fine-tunear el modelo de imágenes con imágenes reales de los programas DMC usando DreamBooth o LoRA de imagen
- Desplegar el pipeline como agente conversacional embebible en dmc.pe (FastAPI + Next.js)
- Evaluar modelos más pequeños (3B) para reducir latencia y costo en producción
