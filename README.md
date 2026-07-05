# hopper-lora-migration

Migrating the [hopper-gen](https://github.com/joshupadhyay) Edward Hopper
style LoRA from **SDXL** (2023, UNet) to **Qwen-Image** (2025, 20B MMDiT),
motivated by Ramp Labs' [PorTAL research](docs/ramp-portal-findings.md) on
porting fine-tunes across base models.

## Why you can't just copy the LoRA over

SDXL LoRA weights are low-rank deltas on SDXL UNet cross-attention layers.
Qwen-Image is an MMDiT transformer with a Qwen2.5-VL text encoder — none of
the target layers exist. Ramp's PorTAL solves this for LLMs with a
hypernetwork (task latents + slim converter), but it's unpublished, LLM-only,
and only pays off across a portfolio of tasks. See
[docs/ramp-portal-findings.md](docs/ramp-portal-findings.md).

**The transferable asset is the task, not the weights:** 27 captioned Hopper
paintings + `hopper style painting` trigger phrase, already sitting in the
`hopper-training-data` Modal volume. Migration = refit onto the new base.

## Model selection

Requirements: newer than SDXL, LoRA-trainable with diffusers, accessible with
the current HF token.

| Model | Arch | Access | Verdict |
|-------|------|--------|---------|
| FLUX.1-dev | 12B DiT | **403 — gated**, license not accepted | blocked (accept license to unblock) |
| SD3.5 medium/large | MMDiT | **403 — gated** | blocked |
| FLUX.1-schnell | 12B DiT (distilled) | open | poor LoRA training target (step-distilled) |
| **Qwen/Qwen-Image** | 20B MMDiT | **open (Apache 2.0)** | ✅ chosen |

## Stack

- **Base model:** Qwen/Qwen-Image
- **Training:** official diffusers `train_dreambooth_lora_qwen_image.py`
  (vendored in `vendor/`), rank 16, lr 1e-4 constant, bf16, cached latents,
  8-bit Adam, gradient checkpointing
- **Compute:** Modal H100 (80 GB)
- **Data:** reuses the `hopper-training-data` Modal volume from hopper-gen

## Usage

```bash
# smoke test (10 steps end-to-end)
modal run modal_train.py --run-name qwen-smoke --max-train-steps 10

# full training run (~600 optimizer steps, checkpoints every 150)
modal run modal_train.py --run-name qwen-v1

# generate samples
modal run modal_generate.py --run-name qwen-v1
modal volume get hopper-training-data outputs-qwen/qwen-v1 ./samples/
```

Requires the `huggingface-secret` Modal secret (HF_TOKEN).
