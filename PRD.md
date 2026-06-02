# Especificaciones e Instrucciones para el Proyecto Final
**Programa:** Diploma AI Engineer  
**Módulo:** Ingeniería de aplicaciones generativas

## 1. Objetivo de la Evaluación
Desarrollar un pipeline generativo multimodal aplicando conceptos de IA Generativa, incluyendo fine-tuning eficiente de modelos de lenguaje (LLM) y generación de contenido visual con modelos de difusión. El proyecto debe demostrar integración funcional entre generación de texto personalizado, producción de imágenes automatizada y documentación técnica orientada a casos de uso empresariales.

## 2. Características del Producto Tecnológico
- Implementación de un pipeline de generación de texto personalizado con fine-tuning sobre un modelo de Hugging Face.
- Aplicación de técnicas PEFT (LoRA o QLoRA) con Unsloth para optimizar el entrenamiento en recursos limitados.
- Sistema de generación de imágenes integrado usando FLUX.1, Stable Diffusion u otro modelo de difusión vía Diffusers.
- Análisis comparativo entre modelo base y modelo fine-tuned con métricas cuantitativas y ejemplos cualitativos.
- Presentación ejecutiva orientada a valor de negocio, con estimación de ROI o impacto.

## 3. Requisitos Técnicos
- Lenguaje principal: Python.
- Uso obligatorio de la biblioteca Transformers de Hugging Face.
- Uso obligatorio de Unsloth con LoRA o Full Fine-tuning sobre modelos de 4B–13B parámetros.
- Uso obligatorio de Diffusers para la generación de imágenes.
- Dataset de mínimo 200 ejemplos relevantes al caso de uso (formato JSON/CSV).
- Código comentado y estructurado, reproducible ejecutando las celdas secuencialmente.
- README con instrucciones de instalación y ejecución.
- Diagrama de arquitectura del sistema.

## 4. Desarrollo del Proyecto
- Explorar arquitecturas de IA Generativa y experimentar con prompt engineering sobre modelos pre-entrenados de Hugging Face.
- Implementar fine-tuning eficiente con Unsloth y LoRA. Preparar dataset, configurar hiperparámetros y presentar demo básica del modelo especializado.
- Integrar el pipeline de generación de imágenes con Diffusers y demostrar el flujo completo texto→imagen.
- Presentar el pipeline completo e integrado, incluyendo comparativa de resultados, estimación de ROI y presentación ejecutiva.

## 5. Entregables Finales
- Repositorio GitHub público o compartido con el docente.
- Código fuente completo del proyecto.
- Notebook funcional en Google Colab o Jupyter Notebook (documentado y reproducible).
- README con pasos de ejecución.
- Documentación técnica y diagrama de arquitectura.
- Diapositivas de presentación final (PDF o PPT, máximo 10 slides).

## 6. Estructura de las Diapositivas de Presentación
La presentación debe ser clara, técnica y orientada a demostrar el pipeline generativo desarrollado y su valor para el negocio. Las diapositivas deben incluir los siguientes apartados:
- Problema a resolver.
- Descripción detallada del problema.
- Justificación de la elección del problema.
- Propuesta de solución planteada.
- Arquitectura del sistema.
- Herramientas y tecnologías utilizadas.
- Conclusiones y posibles mejoras futuras.

## 7. Recomendaciones Finales
- Mantener una estructura clara y modular del código.
- Experimentar con distintos valores de hiperparámetros (learning rate, LoRA r/alpha, guidance_scale) y documentar los resultados.
- Documentar adecuadamente las decisiones de diseño y los experimentos realizados.
- Preparar una demo funcional que muestre el pipeline completo para la presentación final.
- Verificar que todas las dependencias estén incluidas en el proyecto y que el notebook sea reproducible.
