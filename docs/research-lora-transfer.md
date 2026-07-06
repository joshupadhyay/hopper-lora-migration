# Research: Porting LoRA Adapters Across Base Models
## Ramp "PorTAL" + prior art, with a gap analysis for a diffusion-model equivalent

*Compiled 2026-07-05. All claims cited; unverifiable items flagged explicitly.*

---

## 1. Track 1 — Ramp Labs "PorTAL" (Portable Task Adapters for LLMs)

### 1.1 What exists publicly

| Artifact | URL | Status |
|---|---|---|
| X announcement thread (@RampLabs) | https://x.com/RampLabs/status/2072383318516187380 | **Could not fetch directly** (x.com blocks unauthenticated fetch; ThreadReader had no unroll; xcancel mirror returned 503). Content reconstructed from search snippets + Digg coverage. |
| X long-form article "PorTAL: Portable Task Adapters for LLMs" | https://x.com/RampLabs/article/2072381992285647280 | **Paywalled/blocked** — fetch returned HTTP 402. Full method details (architecture diagrams, exact calibration counts, per-task tables) are in here and were NOT retrievable. |
| Karim Atiyeh (Ramp co-founder/CTO) quote-post | https://x.com/karimatiyeh/status/2072385423104491868 | Snippet only (via search): "The trap with fine-tuning has always been that the learning lives inside one model's weights. With PorTAL, we pulled the task representation out of the base entirely. Learn it once, and porting to a new model is just refitting a thin converter, no retraining from scratch." |
| Digg tech aggregation | https://digg.com/tech/7eadbmlh | Fetched successfully; main secondary source. Dated ~Jul 1 2026, describes the announcement as ~4 days prior — note this conflicts with the tweet-ID-implied ~Dec 2026 date in the task brief; the Digg page's framing suggests the announcement is recent as of early July 2026. Treat exact date as unverified. |
| engineering.ramp.com / builders.ramp.com blog post | — | **Does not exist** (site-restricted search found no PorTAL post as of 2026-07-05). |
| arXiv paper, code repo, weights | — | **None found.** Digg explicitly: "the announcement supplies no repo, weights." |
| Hacker News / Reddit threads | — | None found via search as of 2026-07-05. |

### 1.2 Method (as much as is public)

Sources: [Digg coverage](https://digg.com/tech/7eadbmlh), search snippets of the [X thread](https://x.com/RampLabs/status/2072383318516187380).

- **Framing:** "A novel recipe to cheaply port fine-tuning between models. Matches per-task LoRA accuracy at half the cost." Motivation: "At Ramp, every new model release used to mean retraining their fine-tunes from scratch."
- **Architecture:** A **hypernetwork setup** that "learns **task latents** once on a source model and then ports the adaptation to fresh base LLMs by training only a **slim converter** on limited calibration data, sidestepping full LoRA retraining each time a new model appears."
  - Interpretation: the task is represented as a base-model-agnostic latent vector/code; a per-target-model converter maps task latents → adapter weights (or adapter deltas) for that target. Only the converter is (re)trained per new base model.
  - **Not public:** hypernetwork layer structure, latent dimensionality, whether the converter outputs LoRA matrices directly or modulates them, which layers are adapted, rank used.
- **Calibration data:** "limited calibration data"; quantitatively, porting used "**roughly half the calibration examples**" of a from-scratch LoRA. Absolute example counts **not public**.
- **"Even across model families"** is claimed for the porting step.

### 1.3 Models, tasks, metrics

- **Source/target models:** experiments on **Qwen3 variants** (porting between them), plus a hold-out transfer to an **unseen Gemma-3 model**. ([Digg](https://digg.com/tech/7eadbmlh))
- **Sizes:** nothing above **8B** reported ("no results on models larger than 8B"). Exact variant sizes not public.
- **Metric:** ported adapters recovered **94–98% of the usual LoRA accuracy lift** (i.e., of the fine-tune's improvement over base, not absolute accuracy) at **~half the cost / half the calibration examples**. Same pattern held on the unseen Gemma-3 target.
- **Tasks:** not specified publicly; Digg notes no results on **instruction-style tasks**, implying evaluation was on classification/extraction-style internal tasks (consistent with Ramp's fintech workloads — receipt/transaction categorization etc. — but that is inference, not verified).

### 1.4 Limitations / open questions (per public coverage)

- No repo, no weights, no paper, no third-party verification ([Digg](https://digg.com/tech/7eadbmlh)).
- No results >8B, none on instruction-style tasks.
- Unknown: how many source (model, task) LoRA pairs the hypernetwork needs to learn good task latents; whether converter training needs labeled task data or just unlabeled calibration prompts; whether it works when tokenizers differ substantially.

**Bottom line:** PorTAL is an announcement-stage industrial result — a learned, data-light (not data-free) hypernetwork/converter approach for LLM task adapters, unverified externally, with the full write-up locked behind an X article that returns HTTP 402.

---

## 2. Track 2 — Prior art: porting LoRA/PEFT across different base models

### 2.1 Summary table

| Method | Transfer achieved | Data needs | Code | Reported quality |
|---|---|---|---|---|
| **X-Adapter** (arXiv 2312.02238, CVPR 2024) | SD1.5 plugins (LoRA, ControlNet) usable *with* SDXL — runs old UNet alongside new one, bridges via mapping layers. Not a weight conversion; both models run at inference. | Trains mapping layers once on new text–image pairs (null-text training); per-LoRA cost is zero afterward | Yes — project page https://showlab.github.io/X-Adapter/ (GitHub: showlab/X-Adapter) | Works as a "universal upgrader"; adds inference cost of running the SD1.5 UNet; quality metrics qualitative in abstract |
| **LoRA-X** (arXiv 2501.16559, ICLR 2025, Qualcomm AI Research) | Training-free, **data-free** LoRA weight transfer via subspace projection; demonstrated **SD1.5 ↔ SDXL** (both UNet). Applies adapter only to target layers with sufficient subspace similarity | None (weights of both bases only) | No public repo found (checked 2026-07-05; OpenReview: https://openreview.net/forum?id=6cQ6cBqzV3) | "Extensive experiments demonstrate effectiveness"; restricted to same-family/similar architectures because it needs layer-wise subspace similarity |
| **Trans-LoRA** (arXiv 2405.17258, NeurIPS 2024, IBM/MIT) | LoRA re-training on target using **synthetic data** generated by an LLM + a discriminator filter approximating the original data distribution. Within-family (Llama, Gemma) **and cross-family**, and across PEFT types | Nearly data-free w.r.t. *original* data (small seed to steer the synthetic curriculum); substantial synthetic-data generation + full re-training compute | No public repo found (IBM publication page: https://research.ibm.com/publications/trans-lora-towards-data-free-transferable-parameter-efficient-finetuning) | "Lossless (mostly improved) transfer" — strongest quality claims of the group, but it is really automated re-training, not conversion |
| **Cross-LoRA** (arXiv 2508.05232) | Data-free LoRA transfer across **heterogeneous LLM architectures** (dimension mismatch handled): LoRA-Align (rank-truncated SVD + Frobenius-optimal linear map to align subspaces) + LoRA-Shift (project source ΔW into target space) | None; "lightweight adaptation on a commodity GPU in ~20 minutes" | No public repo found | Up to +5.26% relative over target base on ARC/OBQA/HellaSwag; "comparable to directly trained LoRA" on commonsense tasks — evaluated on LLMs only |
| **ProLoRA** (arXiv 2506.04244, ICML 2025) — "Zero-Shot Adaptation of PEFT in Diffusion Models" | Training-free, data-free projection of LoRA between **text-to-image diffusion models** using subspace + null-space similarity, selectively targeting aligned layers | None | No public repo found | "Successful knowledge transfer and comparable performance without retraining"; specific model pairs not stated in abstract — likely SD-family UNet→UNet (unverified; check full PDF) |
| **Text-to-LoRA (T2L)** (arXiv 2506.06105, ICML 2025, Sakana AI) | Hypernetwork generates a LoRA for a *fixed* base LLM from a text description of the task, in one forward pass; zero-shot to unseen tasks. **Not cross-model** — but the closest public blueprint for "hypernetwork emits LoRA weights" | Trained by distilling **9 pre-trained task LoRAs** (GSM8K, ARC, etc.) or multi-task SFT | **Yes** — https://github.com/SakanaAI/text-to-lora | Matches task-specific LoRA performance on trained tasks; degrades gracefully zero-shot |
| **HyperDreamBooth** (arXiv 2307.06949, CVPR 2024, Google) | Hypernetwork (ViT encoder + transformer decoder) predicts lightweight LoRA-like weights (LiDB, ~100KB) for SD from **one face image**; + rank-relaxed fast fine-tune. Fixed base model | Trained on a face dataset; per-subject personalization ~20s (25× faster than DreamBooth) | Paper: https://arxiv.org/abs/2307.06949 (official code not released; community reimplementations exist) | DreamBooth-level fidelity on faces; proves hypernetwork→diffusion-LoRA generation is viable |
| **DiffLoRA** (arXiv 2408.06740) | Diffusion model *generates* personalized LoRA weights (weights-as-data), fixed base | Trained on many (image, LoRA) pairs | See paper page | Same category as HyperDreamBooth: weight generation, not cross-model porting |

### 2.2 Community / practitioner state: SDXL → FLUX / SD3

- **No true converter exists.** Consensus in ComfyUI/tutorial ecosystem: LoRAs only work on the base they were trained on; SDXL (2.6B UNet) vs FLUX-dev (12B rectified-flow MMDiT) differ in architecture, dimensionality, text encoders, and objective, so direct weight conversion is not possible ([TechXplainator ComfyUI guide](https://techxplainator.com/comfyui-8-using-loras/), [MimicPC guide](https://www.mimicpc.com/learn/add-lora-in-comfyui)).
- **"Workarounds" are pipeline tricks, not conversions:** e.g., run an SD1.5/SDXL pass with the LoRA, then refine/img2img with FLUX ([OpenArt workflow](https://openart.ai/workflows/cgtips/comfyui---elevate-flux-performance-with-sdsdxl-lora-models/UOUIY9T4gT2Cu0fovy6a), [Code Crafters Corner](https://www.patreon.com/posts/sd-1-5-sdxl-with-112357793)) — conceptually a poor man's X-Adapter.
- The other direction exists as *retraining*: e.g., a "FLUX-style LoRA for SDXL" trained from scratch on FLUX outputs ([Civitai](https://civitai.com/models/625636/flux-style-lora-for-sdxl)) — style distillation via images, not weight transfer.
- LoRA behavior itself differs across the divide: MMDiT LoRAs train stably at higher effective rank without the initialization tricks UNet SDXL sometimes needs (noted in T-LoRA, https://arxiv.org/pdf/2507.05964). Merging across UNet/DiT is discussed as impossible-in-principle in community write-ups ([Civitai article on FLUX UNet/DiT merging](https://civitai.com/articles/3409/flux-diagram-of-unet-dit-and-exotic-merging-methods-v801)).
- I found **no ai-toolkit (ostris) or ComfyUI GitHub issue proposing an actual UNet→MMDiT LoRA weight-mapping tool** — searches surfaced only usage guides and pipeline workarounds. (Absence of evidence, flagged as such; a targeted GitHub-issues crawl could still turn something up.)

---

## 3. Gap analysis — what does NOT exist

1. **Cross-architecture (UNet → DiT/MMDiT) LoRA transfer for diffusion models.** Nothing published or shipped:
   - X-Adapter: same-lineage upgrade (SD1.5→SDXL, both UNet), and it *keeps the old model running* rather than converting weights.
   - LoRA-X and ProLoRA: training-free projections that depend on **layer-wise subspace similarity** — by construction they need architecturally comparable layers. SD1.5↔SDXL works; UNet→FLUX/SD3 has no layer correspondence to exploit (conv/attention UNet blocks vs. double/single-stream MMDiT blocks; different text conditioning: CLIP vs. T5+CLIP).
   - Cross-LoRA handles heterogeneous *LLMs* (still homogeneous in kind: stacks of transformer blocks with attention/MLP). No diffusion version.
2. **A learned converter/hypernetwork (PorTAL-style) for diffusion adapters — in any direction.** PorTAL itself is LLM-only, closed, unverified. T2L/HyperDreamBooth generate LoRAs but always for one fixed base. Nobody has published "task/style latents + per-base converter" for image models.
3. **Style LoRAs specifically:** all data-based transfer work (Trans-LoRA) is LLM/task-accuracy oriented. Style is easier in one way (the "training data" can be regenerated: sample images from source-model+LoRA) and harder in another (no scalar metric; quality = human/CLIP/DINO similarity judgments). No paper exploits the regeneration trick systematically as a *general SDXL→FLUX porting service*, even though it is the obvious Trans-LoRA analogue.
4. **No public N-pairs dataset of "same concept trained on two bases"** (e.g., 500 styles each trained as SDXL LoRA and FLUX LoRA) that a converter could be supervised on. This dataset does not exist publicly and would itself be a contribution.

---

## 4. Feasibility notes — building "PorTAL for image models" (SDXL → FLUX/SD3/Qwen-Image)

### 4.1 Three viable routes, ordered by risk

**Route A — Trans-LoRA analogue (regenerate-and-retrain). Lowest risk, known to work in principle.**
- For a style LoRA, the original data is replaceable: sample K images from (source base + LoRA), optionally caption with a VLM, then train a target-model LoRA on those samples. Add a filter (CLIP/DINO scoring vs. LoRA-off baselines) as the Trans-LoRA "discriminator."
- Data: 50–500 generated images per LoRA. Compute per LoRA: ~minutes of SDXL sampling + a standard FLUX LoRA run (~1–3 h on one A100/H100 with ai-toolkit-style configs). This is "automated retraining," not a converter — but it is the baseline any learned method must beat on cost.

**Route B — PorTAL-style learned converter (hypernetwork). The interesting one.**
- Supervision: build N (source-LoRA, target-LoRA) pairs by running Route A at scale. Train a converter f(θ_src_LoRA) → θ_tgt_LoRA, or better, T2L-style: encode each LoRA into a latent (SVD-compress ΔW per layer → tokens), decode into target layer set.
- How many pairs? T2L got zero-shot generalization from only **9** task LoRAs on LLMs (https://arxiv.org/abs/2506.06105); style space is much broader — expect **hundreds to low thousands** of style pairs for useful generalization. HyperDreamBooth needed a large face dataset but its output space was tiny (~100KB LiDB).
- Compute estimate: N=500 pairs × (Route A cost ≈ 1–2 GPU-h) ≈ **~500–1,000 GPU-hours to build the training set** (the dominant cost), then converter training itself is cheap (the converter is an MLP/transformer over weight tokens; days on 1–8 GPUs). At ~$2–3/H100-hr on Modal, dataset ≈ **$1–3k**, converter training ≈ **$100–500**. Per-port inference afterward: seconds, plus optional "slim refit" on a handful of generated calibration images (this is exactly PorTAL's ~half-calibration trick, transplanted).
- Closest reusable code: **Text-to-LoRA** (https://github.com/SakanaAI/text-to-lora) for the hypernetwork/weight-tokenization scaffolding; ai-toolkit/kohya for mass LoRA training; Trans-LoRA's synthetic-curriculum idea for the calibration step.

**Route C — training-free projection (LoRA-X/Cross-LoRA extended to UNet→MMDiT). Highest risk.**
- Would require inventing a cross-modal layer correspondence (e.g., match SDXL cross-attention K/V in text-conditioned blocks to FLUX double-stream text-stream projections via Cross-LoRA-style SVD alignment). No prior success; subspace-similarity assumptions likely break across UNet/MMDiT. Worth a 1-week probe at most; publishable if it works even partially.

### 4.2 Recommended framing

Route A as the product baseline + Route B as the differentiated "PorTAL for image models" bet. The dataset of paired LoRAs (Route A at scale) de-risks Route B and is independently valuable. Key open technical choices mirror PorTAL's undisclosed ones: latent representation of a LoRA (per-layer SVD tokens is the current standard), whether the converter is per-target-model (PorTAL's answer: yes, a "slim converter" per new base), and how much calibration refit to allow (PorTAL: ~half of from-scratch examples, recovering 94–98% of lift).

---

## Sources

**PorTAL / Ramp**
- Ramp Labs X thread: https://x.com/RampLabs/status/2072383318516187380 *(not directly fetchable)*
- Ramp Labs X article "PorTAL: Portable Task Adapters for LLMs": https://x.com/RampLabs/article/2072381992285647280 *(HTTP 402, not retrievable)*
- Karim Atiyeh post: https://x.com/karimatiyeh/status/2072385423104491868 *(snippet via search only)*
- Digg coverage: https://digg.com/tech/7eadbmlh
- Ramp engineering blog (no PorTAL post found): https://engineering.ramp.com/ , https://builders.ramp.com/

**Papers**
- X-Adapter: https://arxiv.org/abs/2312.02238 ; project: https://showlab.github.io/X-Adapter/
- LoRA-X (ICLR 2025): https://arxiv.org/abs/2501.16559 ; OpenReview: https://openreview.net/forum?id=6cQ6cBqzV3
- Trans-LoRA (NeurIPS 2024): https://arxiv.org/abs/2405.17258 ; IBM page: https://research.ibm.com/publications/trans-lora-towards-data-free-transferable-parameter-efficient-finetuning ; IBM blog: https://research.ibm.com/blog/LoRAs-explained
- Cross-LoRA: https://arxiv.org/abs/2508.05232
- ProLoRA (ICML 2025): https://arxiv.org/abs/2506.04244
- Text-to-LoRA (ICML 2025): https://arxiv.org/abs/2506.06105 ; code: https://github.com/SakanaAI/text-to-lora ; coverage: https://www.marktechpost.com/2025/06/13/sakana-ai-introduces-text-to-lora-t2l-a-hypernetwork-that-generates-task-specific-llm-adapters-loras-based-on-a-text-description-of-the-task/
- HyperDreamBooth: https://arxiv.org/abs/2307.06949
- DiffLoRA: https://arxiv.org/html/2408.06740
- T-LoRA (MMDiT LoRA rank behavior): https://arxiv.org/pdf/2507.05964

**Community (SDXL↔FLUX)**
- ComfyUI LoRA guide (base-model coupling): https://techxplainator.com/comfyui-8-using-loras/
- MimicPC guide: https://www.mimicpc.com/learn/add-lora-in-comfyui
- OpenArt SD/SDXL-LoRA-with-FLUX workflow (workaround, not conversion): https://openart.ai/workflows/cgtips/comfyui---elevate-flux-performance-with-sdsdxl-lora-models/UOUIY9T4gT2Cu0fovy6a
- SD1.5/SDXL LoRAs with FLUX (pipeline workaround): https://www.patreon.com/posts/sd-1-5-sdxl-with-112357793
- FLUX-style LoRA for SDXL (retrained, not converted): https://civitai.com/models/625636/flux-style-lora-for-sdxl
- FLUX UNet/DiT merging discussion: https://civitai.com/articles/3409/flux-diagram-of-unet-dit-and-exotic-merging-methods-v801

**Verification caveats**
- PorTAL details beyond the Digg summary and search snippets are unverified; the full X article is inaccessible.
- ProLoRA's exact model pairs, and code availability for LoRA-X / Trans-LoRA / Cross-LoRA / ProLoRA, could not be confirmed from abstracts as of 2026-07-05.
- Announcement date discrepancy: task brief says ~Dec 2026 (tweet ID), Digg framing implies ~late June 2026. Unresolved.
