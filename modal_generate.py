"""
Generate Hopper-style images with the migrated Qwen-Image LoRA.

Usage:
    modal run modal_generate.py --run-name qwen-v1
    modal run modal_generate.py --run-name qwen-v1 --checkpoint checkpoint-300
"""

import modal

app = modal.App("hopper-qwen-lora-generate")

training_data = modal.Volume.from_name("hopper-training-data")
model_cache = modal.Volume.from_name("hopper-model-cache")

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"
MODEL_ID = "Qwen/Qwen-Image"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "peft>=0.14.0",
        "bitsandbytes",
        "Pillow",
        "safetensors",
        "sentencepiece",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

PROMPTS = [
    "hopper style painting, a lone man in a diner at night, harsh fluorescent light, empty street visible through the window, muted greens and deep shadows",
    "hopper style painting, woman sitting on a bed facing a bright window, morning sunlight cutting across the wall, sparse room, quiet isolation",
    "hopper style painting, gas station at dusk on an empty rural road, glowing red pumps, dark pine forest behind, last light on the horizon",
    "hopper style painting, an usher standing alone in a movie theater aisle, dim wall sconces, red curtains, lost in thought",
    "hopper style painting, sunlight on the side of a white lighthouse, stark geometric shadows, blue sky, coastal grass",
    "hopper style painting, a self-driving car charging station at night, lone attendant under artificial light, empty highway beyond",
]


@app.function(
    image=image,
    gpu="H100",  # bf16 inference; pass --quantize for the 24GB-GPU NF4 path
    timeout=7200,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(
    run_name: str = "qwen-v1",
    checkpoint: str = "",
    lora_scale: float = 1.0,
    num_images: int = 0,
    quantize: bool = False,
):
    import torch
    from pathlib import Path
    from diffusers import QwenImagePipeline

    adapter_dir = Path(DATA_DIR) / "adapters-qwen" / run_name
    if checkpoint:
        adapter_dir = adapter_dir / checkpoint
    print(f"Loading LoRA from {adapter_dir}")

    quant_config = None
    if quantize:
        from diffusers.quantizers import PipelineQuantizationConfig

        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )
    pipe = QwenImagePipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, quantization_config=quant_config
    )
    pipe.load_lora_weights(str(adapter_dir))
    pipe.to("cuda")

    out_dir = Path(DATA_DIR) / "outputs-qwen" / (run_name + (f"-{checkpoint}" if checkpoint else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator("cuda").manual_seed(42)
    prompts = PROMPTS[:num_images] if num_images else PROMPTS
    for i, prompt in enumerate(prompts):
        img = pipe(
            prompt,
            width=1024,
            height=1024,
            num_inference_steps=40,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            generator=generator,
            attention_kwargs={"scale": lora_scale},
        ).images[0]
        fname = out_dir / f"sample_{i}.png"
        img.save(fname)
        print(f"Saved {fname}")

    training_data.commit()
    return {"output_dir": str(out_dir), "num_images": len(prompts)}


@app.local_entrypoint()
def main(
    run_name: str = "qwen-v1",
    checkpoint: str = "",
    lora_scale: float = 1.0,
    num_images: int = 0,
    quantize: bool = False,
):
    result = generate.remote(run_name, checkpoint, lora_scale, num_images, quantize)
    print(result)
    print(f"\nDownload with: modal volume get hopper-training-data {result['output_dir'].removeprefix('/data/')} ./samples/")
