from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import torch
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

# Modello
MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
VAE_PATH = "models/vae-ft-mse/vae.pt"

# Impostazioni
width = 896
height = 896
num_images = 20
num_steps = 30
guidance = 5.5

# Negative prompt
negative_prompt = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), "
    "text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
    "mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, "
    "dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, "
    "missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
)

# Carica i prompt
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Output
model_name = MODEL_ID.split("/")[-1]
output_dir = os.path.join("outputs", model_name)
os.makedirs(output_dir, exist_ok=True)

log_file = f"completed_{model_name}.txt"
completed = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed = set(line.strip() for line in f)

# Caricamento pipeline
print("🔧 Caricamento VAE e pipeline...")
vae = AutoencoderKL.from_single_file(VAE_PATH, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Generazione
for prompt_index, prompt in enumerate(tqdm(prompts, desc=model_name), start=1):
    for image_index in range(1, num_images + 1):
        filename = f"{prompt_index}_{model_name}_{image_index}.png"
        save_path = os.path.join(output_dir, filename)

        retry_attempts = 5  # massimo 5 tentativi

        while retry_attempts > 0:
            if filename in completed and os.path.exists(save_path):
                img = Image.open(save_path)
                img_array = np.array(img)
                if img_array.mean() > 2:
                    break  # immagine buona, passa oltre

            print(f"⚠️ Rilevata immagine nera: rigenero {filename} ({6 - retry_attempts}/5)")

            # Rigenera
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=width,
                height=height
            ).images[0]

            image.save(save_path)

            # Se rigenerata correttamente, salva e esci dal ciclo
            img_array = np.array(image)
            if img_array.mean() > 2:
                with open(log_file, "a") as logf:
                    logf.write(filename + "\n")
                print(f"✅ Salvata: {filename}")
                break

            retry_attempts -= 1

        if retry_attempts == 0:
            print(f"❌ Errore: {filename} non è stato possibile rigenerarlo senza NSFW dopo 5 tentativi.")

