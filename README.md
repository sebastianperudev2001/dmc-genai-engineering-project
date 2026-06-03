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
| `03_pipeline_integrado.ipynb` | Pipeline end-to-end integrado: mensaje del usuario → clasificación de intención → recomendación → imagen del curso + comparativa de resultados + estimación de ROI |

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
- GPU T4 (mínimo) o A100 (recomendado para Plan B)

### Pasos

**1. Generar el dataset (local — no requiere GPU):**
```bash
cd notebooks/dataset
python generate_dataset.py
# Genera: dmc_leads.json (200 ejemplos)
```

**2. Subir a Colab y ejecutar en orden:**
```
00_exploracion.ipynb      → no requiere outputs previos
01_finetuning.ipynb       → requiere dmc_leads.json subido a /content/
02_image_generation.ipynb → no requiere outputs previos
03_pipeline_integrado.ipynb → requiere outputs/lora_adapter/ y image_generator.py
```

---

## Conclusiones y Posibles Mejoras Futuras

### Conclusiones
- El fine-tuning con Unsloth/LoRA sobre un dataset de dominio específico debería permitir al modelo clasificar motivaciones con mayor precisión que el prompting estándar, reduciendo errores de formato y alucinaciones
- La integración de un generador de imágenes (Diffusers) enriquece la recomendación de cursos con contenido visual personalizado, agregando una dimensión multimodal al pipeline
- El caso de uso de DMC Institute permite validar el pipeline en un contexto de negocio real, con métricas concretas de conversión y ROI estimable

### Mejoras Futuras
- Aumentar el dataset a 1000+ ejemplos con conversaciones multi-turno para mejorar el comportamiento conversacional del agente
- Integrar RAG con los brochures reales de DMC (S3 + embeddings) para que las respuestas estén fundamentadas en documentos oficiales
- Fine-tunear el modelo de imágenes con imágenes reales de los programas DMC usando DreamBooth
- Desplegar el pipeline como agente conversacional embebible en dmc.pe (FastAPI + Next.js)
- Evaluar modelos más pequeños (3B) para reducir latencia y costo en producción
