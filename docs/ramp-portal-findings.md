# Ramp Labs "PorTAL" — Key Ideas & What They Mean for This Migration

*Source: [Ramp Labs announcement on X](https://x.com/RampLabs/status/2072383318516187380) (Dec 2026 thread, surfaced via [this tweet](https://x.com/escarghoe_/status/2072509683957407815)); summary coverage on [Digg](https://digg.com/tech/7eadbmlh). No public paper, code, or weights as of Jul 2026.*

## The problem PorTAL solves

Ramp fine-tunes LLMs per task (LoRA adapters). Every new base-model release
meant retraining every fine-tune from scratch — the "switching overhead" of
model upgrades. LoRA weights are deltas *relative to a specific base model's
weights*; they are meaningless applied to a different base.

## How PorTAL works

**PorTAL = Portable Task Adapters for LLMs.** Three pieces:

1. **Task latents** — a hypernetwork is trained once on a *source* model to
   compress what a task's adaptation "is" into a compact latent vector,
   decoupled from any specific base model's weight space.
2. **Slim converter** — to port to a *new* base model, only a small converter
   network is trained (task latent → LoRA weights in the new model's weight
   space), using **limited calibration data** — roughly half the examples a
   full LoRA retrain needs.
3. **Cross-family transfer** — reported to work even across model families
   (trained on Qwen3 variants, transferred to unseen Gemma-3 models).

**Reported results:** recovers 94–98% of the full per-task LoRA accuracy lift
at ~half the cost.

## Caveats

- Announcement-stage: no public code, weights, or paper.
- Validated only on ≤8B LLMs; no results on instruction-style tasks, larger
  models, or third-party reproduction.
- **LLM-only.** Nothing published about diffusion/image models.

## Can PorTAL migrate an image LoRA? (the honest answer)

**Not literally, today:**

- No public implementation exists to run.
- The hypernetwork must be trained on the source model per task — for a
  *single* task (one style LoRA) that costs more than just retraining. PorTAL
  amortizes across many tasks × many model upgrades; Ramp has a portfolio of
  fine-tunes, we have one.
- Diffusion architectures diverge harder than LLM families: SDXL is a
  **UNet** conditioned on CLIP embeddings; every serious post-SDXL model
  (FLUX, SD3.5, Qwen-Image) is a **DiT/MMDiT transformer** with different
  text encoders (T5, Qwen2.5-VL). There is no layer-to-layer correspondence
  for a converter to exploit — the SDXL LoRA targets `UNet` cross-attention
  blocks that simply don't exist in an MMDiT.
- Closest image-world analogues: X-Adapter (bolt-on adapter making SD1.5
  LoRAs drive SDXL) and zero-shot PEFT-adaptation work for diffusion models
  (e.g. [arXiv 2506.04244](https://arxiv.org/pdf/2506.04244)) — research-grade,
  not drop-in.

**But the *idea* transfers, and it's the design principle of this repo:**

> The durable asset is the **task**, not the adapter weights.

For hopper-gen, the "task latent" is effectively the dataset: 27 curated
Hopper paintings + hand-written captions + the `hopper style painting`
trigger phrase + a tuned training recipe. Migration = refit that task onto
the new base model. That's exactly what PorTAL operationalizes for LLMs, and
what we do here manually: same data volume, same captions, new base
(Qwen-Image), fresh LoRA.

## Practical implications for anyone maintaining fine-tunes

1. **Treat datasets + eval prompts as the long-lived artifact.** Version them
   like code; adapters are disposable build outputs.
2. **Expect base-model churn.** Budget for periodic refits; keep training
   scripts parameterized by base model.
3. **Calibration-data efficiency is the frontier.** PorTAL's real win is
   halving the data/compute per refit — watch for an open-source
   implementation or reproduction before adopting.
