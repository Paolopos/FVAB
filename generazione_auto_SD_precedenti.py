from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import torch
import os
from tqdm import tqdm

# Modelli da usare
MODELS = [
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
]

# Scheduler specifici per alcuni modelli
SCHEDULERS = {
    "stabilityai/stable-diffusion-2": EulerDiscreteScheduler,
    "stabilityai/stable-diffusion-2-base": EulerDiscreteScheduler,
    "stabilityai/stable-diffusion-2-1": DPMSolverMultistepScheduler,
    "stabilityai/stable-diffusion-2-1-base": EulerDiscreteScheduler,
}

# Parametri generali
width = 512
height = 512
num_images = 20
num_steps = 25
guidance = 5.5

# Nessun negative_prompt ufficiale specificato nei readme
negative_prompt = ""

# Carica i prompt
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Loop sui modelli
for model_id in MODELS:
    model_name = model_id.split("/")[-1]
    print(f"\n📦 Avvio generazione per: {model_name}")

    # Output directory
    model_output_dir = os.path.join("outputs", model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Log file
    log_file = f"completed_{model_name}.txt"
    completed = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            completed = set(line.strip() for line in f)

    # Caricamento pipeline
    print("🔧 Caricamento pipeline...")
    if model_id in SCHEDULERS:
        scheduler = SCHEDULERS[model_id].from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None
        )

    pipe.to("cuda")

    # Generazione immagini
    for prompt_index, prompt in enumerate(tqdm(prompts, desc=model_name), start=1):
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