from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
import os
from tqdm import tqdm

# Modelli da usare
MODELS = [
    "SG161222/Realistic_Vision_V1.3",
    "SG161222/Realistic_Vision_V1.4",
    "SG161222/Realistic_Vision_V2.0",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "SG161222/Realistic_Vision_V4.0_noVAE",
    "SG161222/Realistic_Vision_V5.0_noVAE",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "SG161222/Realistic_Vision_V6.0_B1_noVAE"
]

# Modelli che richiedono VAE esterno
MODELS_NEED_VAE = {
    "SG161222/Realistic_Vision_V2.0",
    "SG161222/Realistic_Vision_V5.0_noVAE",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "SG161222/Realistic_Vision_V6.0_B1_noVAE"
}

VAE_ID = "stabilityai/sd-vae-ft-mse-original"

# Parametri generali
width = 512
height = 512
num_images = 20
num_steps = 25
guidance = 5.5

# Prompt negativo comune
negative_prompt = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), "
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
)

# Carica i prompt
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Loop sui modelli
for model_id in MODELS:
    model_name = model_id.split("/")[-1]
    print(f"\n📦 Avvio generazione per: {model_name}")

    # Percorso della cartella output per questo modello
    model_output_dir = os.path.join("outputs", model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Log file
    log_file = f"completed_{model_name}.txt"
    completed = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            completed = set(line.strip() for line in f)

    # Carica pipeline con o senza VAE
    if model_id in MODELS_NEED_VAE:
        print("🔧 Caricamento VAE esterno...")
        vae = AutoencoderKL.from_single_file("models/vae-ft-mse/vae.pt", torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )


    pipe.to("cuda")

    # Generazione immagini
    for prompt_index, prompt in enumerate(tqdm(prompts, desc=f"{model_name}"), start=1):
        for image_index in range(1, num_images + 1):
            filename = f"{prompt_index}_{model_name}_{image_index}.png"
            save_path = os.path.join(model_output_dir, filename)

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
