import requests
import uuid

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HF_TOKEN = "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"

def generate_image(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    filename = f"output_{uuid.uuid4().hex}.png"
    
    with open(filename, "wb") as f:
        f.write(response.content)

    return filename

