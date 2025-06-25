from diffusers import StableDiffusionPipeline
import torch 
import os
import uuid

pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR="images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(IMAGE_DIR, filename)
    image.save(path)
    return path