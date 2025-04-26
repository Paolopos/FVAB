from diffusers import StableDiffusionPipeline
import torch
import os
from tqdm import tqdm

# Modello da usare
MODEL_ID = "SG161222/Paragon_V1.0"

# Parametri specifici per Paragon_V1.0
width = 512
height = 512
num_images = 20
num_steps = 25
guidance = 5.5  # Consigliato tra 5 e 12, teniamo 5.5 come standard
clip_skip = 2   # Importante: Paragon consiglia Clip Skip 2 (ma diffusers non supporta nativamente Clip Skip: serve modello addestrato già così, quindi ok)
device = "cuda"

# Prompt negativo consigliato dal readme
negative_prompt = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), "
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, "
    "easynegative, bad-hands-5"
)

# Carica i prompt
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Percorso output
model_name = MODEL_ID.split("/")[-1]
output_dir = os.path.join("outputs", model_name)
os.makedirs(output_dir, exist_ok=True)

# Log file
log_file = f"completed_{model_name}.txt"
completed = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed = set(line.strip() for line in f)

# Carica pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
pipe.to(device)

# Generazione immagini
for prompt_index, prompt in enumerate(tqdm(prompts, desc=f"{model_name}"), start=1):
    for image_index in range(1, num_images + 1):
        filename = f"{prompt_index}_{model_name}_{image_index}.png"
        save_path = os.path.join(output_dir, filename)

        if filename in completed:
            continue

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=width,
            height=height
        ).images[0]

        image.save(save_path)

        with open(log_file, "a") as logf:
            logf.write(filename + "\n")

        print(f"✅ Salvata: {filename}")
