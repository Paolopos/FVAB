import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import os
from tqdm import tqdm

# Config generali
prompt_file = "prompts.txt"
num_images = 20
prior_steps = 20
decoder_steps = 10
prior_guidance = 4.0
decoder_guidance = 0.0
width = 1024
height = 1024
output_dir = "outputs/StableCascade"

# Prompt negativo lasciato vuoto come da documentazione
negative_prompt = ""

# Carica i prompt
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Caricamento pipeline
print("🔧 Caricamento modelli Stable Cascade...")
prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior",
    torch_dtype=torch.bfloat16,
    variant="bf16"
)
decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    torch_dtype=torch.bfloat16,
    variant="bf16"
)

prior.enable_model_cpu_offload()
decoder.enable_model_cpu_offload()

# Log e cartella output
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "completed.txt")
completed = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed = set(line.strip() for line in f)

# Loop generazione
for prompt_index, prompt in enumerate(tqdm(prompts, desc="StableCascade"), start=1):
    for image_index in range(1, num_images + 1):
        filename = f"{prompt_index}_StableCascade_{image_index}.png"
        save_path = os.path.join(output_dir, filename)
        if filename in completed:
            continue

        # Fase 1: prior
        prior_output = prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=prior_guidance,
            num_images_per_prompt=1,
            num_inference_steps=prior_steps
        )

        # Fase 2: decoder
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance,
            output_type="pil",
            num_inference_steps=decoder_steps
        )

        image = decoder_output.images[0]
        image.save(save_path)

        with open(log_file, "a") as logf:
            logf.write(filename + "\n")

        print(f"✅ Salvata: {filename}")
