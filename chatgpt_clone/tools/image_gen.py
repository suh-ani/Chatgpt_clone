from diffusers import StableDiffusionPipeline
import torch
import os
import uuid

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_image(user_prompt):
    # Enhanced prompt with general quality + face-boosting terms
    prompt = (
        f"{user_prompt}, ultra-detailed, 8k resolution, high quality, sharp focus, "
        "cinematic lighting, photorealistic, beautiful face, clear facial features, "
    )

    negative_prompt = (
        "blurry, distorted face, deformed, bad anatomy, fused fingers, low resolution, "
        "out of frame, extra limbs, ugly, poorly drawn face, grainy"
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]

    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(IMAGE_DIR, filename)
    image.save(path)
    return path
