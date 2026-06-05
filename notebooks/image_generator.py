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
