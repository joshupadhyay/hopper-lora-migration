"""
Migrate the hopper-gen Hopper-style LoRA from SDXL to Qwen-Image.

Reuses the existing `hopper-training-data` Modal volume (27 captioned Hopper
paintings) and trains a fresh LoRA on Qwen/Qwen-Image (20B MMDiT, Apache 2.0)
using the official diffusers training script vendored in vendor/.

Per Ramp's PorTAL framing: the durable asset is the task (dataset + captions +
trigger phrase), not the adapter weights. SDXL UNet LoRA weights cannot be
loaded into an MMDiT transformer, so migration = refit the task onto the new
base model.

Usage:
    modal run modal_train.py --run-name qwen-v1 --max-train-steps 10   # smoke test
    modal run modal_train.py --run-name qwen-v1                        # full run
"""

import modal

app = modal.App("hopper-qwen-lora-train")

training_data = modal.Volume.from_name("hopper-training-data", create_if_missing=True)
model_cache = modal.Volume.from_name("hopper-model-cache", create_if_missing=True)

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

MODEL_ID = "Qwen/Qwen-Image"
TRAIN_SCRIPT = "/root/train_dreambooth_lora_qwen_image.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "peft>=0.14.0",
        "bitsandbytes",
        "datasets",
        "Pillow",
        "safetensors",
        "sentencepiece",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("vendor/train_dreambooth_lora_qwen_image.py", TRAIN_SCRIPT)
)


@app.function(
    image=image,
    gpu="A10G",  # account is limited to 24GB GPUs (no payment method on file);
    # the 20B base model is NF4-quantized to fit (QLoRA)
    timeout=36000,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    run_name: str = "qwen-v1",
    max_train_steps: int = 600,
    rank: int = 16,
    learning_rate: float = 1e-4,
    checkpointing_steps: int = 150,
):
    import json
    import shutil
    import subprocess
    from pathlib import Path

    # --- Build an imagefolder dataset from the existing hopper-gen data ---
    # data/processed/*.jpg + data/captions.jsonl (file_name/text) -> the exact
    # metadata.jsonl format HF datasets' imagefolder loader expects.
    data_path = Path(DATA_DIR)
    dataset_dir = data_path / "qwen_dataset"
    dataset_dir.mkdir(exist_ok=True)

    captions = [json.loads(line) for line in (data_path / "captions.jsonl").open()]
    for entry in captions:
        src = data_path / "processed" / entry["file_name"]
        dst = dataset_dir / entry["file_name"]
        if not dst.exists():
            shutil.copy(src, dst)
    with (dataset_dir / "metadata.jsonl").open("w") as f:
        for entry in captions:
            f.write(json.dumps(entry) + "\n")
    training_data.commit()
    print(f"Dataset ready: {len(captions)} captioned images in {dataset_dir}")

    output_dir = f"{DATA_DIR}/adapters-qwen/{run_name}"

    # NF4 quantization config so the 20B transformer fits in 24GB
    bnb_config_path = "/root/bnb_nf4.json"
    with open(bnb_config_path, "w") as f:
        json.dump(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
            f,
        )

    cmd = [
        "accelerate", "launch", "--num_processes=1", "--mixed_precision=bf16",
        TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", MODEL_ID,
        "--dataset_name", str(dataset_dir),
        "--caption_column", "text",
        "--instance_prompt", "hopper style painting",
        "--output_dir", output_dir,
        "--mixed_precision", "bf16",
        "--resolution", "1024",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--rank", str(rank),
        "--lora_alpha", str(rank),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_train_steps),
        "--checkpointing_steps", str(checkpointing_steps),
        "--bnb_quantization_config_path", bnb_config_path,
        "--use_8bit_adam",
        "--gradient_checkpointing",
        "--cache_latents",
        # no --offload: everything is GPU-resident (quantized TE + transformer);
        # offload's .to() calls are unsupported on bnb-quantized models
        "--seed", "42",
    ]
    print("Launching:", " ".join(cmd))
    result = subprocess.run(cmd)
    training_data.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    saved = list(Path(output_dir).rglob("*.safetensors"))
    print(f"Saved adapter files: {[str(p) for p in saved]}")
    return {"run_name": run_name, "steps": max_train_steps, "output_dir": output_dir}


@app.local_entrypoint()
def main(
    run_name: str = "qwen-v1",
    max_train_steps: int = 600,
    rank: int = 16,
    learning_rate: float = 1e-4,
    checkpointing_steps: int = 150,
):
    result = train.remote(run_name, max_train_steps, rank, learning_rate, checkpointing_steps)
    print("\n" + "=" * 40)
    for k, v in result.items():
        print(f"  {k}: {v}")
