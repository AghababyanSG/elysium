# Elysium VLA — Remote Handoff (2026-05-13)

This document is a self-contained handoff so you can resume work on a remote
GPU machine. It captures the project context, the architecture roadmap, what
was done in Phase 1, and the exact prompt to paste into Claude on the remote
machine.

---

## Default prompt to paste into Claude on the remote device

> ```
> I'm continuing work on the Elysium VLA project. Read these files in order
> before doing anything else:
>
>   1. plans/HANDOFF.md  -- this handoff document (the full prior conversation
>      context, what's done, what's next).
>   2. plans/architecture-roadmap.md  -- the phased roadmap with checkboxes;
>      Phase 1 sub-steps 1.1-1.7 are already complete in the code (boxes ticked).
>      Phase 1.8 (full SFT) and 1.9 (RL) remain to be run on this GPU.
>   3. CLAUDE.md  -- the project's persistent code rules.
>
> Current state:
>   - Branch: training-upgrade
>   - Phase 1 coord-token binning is implemented and tested (33 pytest
>     passing). data/processed already contains sentinel-encoded actions.
>   - configs/train.yaml is preset for THIS remote machine: model.use_unsloth
>     = false, model.load_in_4bit = false, batch_size = 4, gradient
>     accumulation = 4, gradient_checkpointing = true.
>   - The Unsloth path is preserved behind use_unsloth=true; do NOT remove it.
>
> What I want next:
>   1. Verify GPU + environment: `nvidia-smi`, `python -m pytest tests/ -q`.
>   2. Run Phase 1.8 SFT on the full dataset:
>        python scripts/train.py --skip-prepare 2>&1 | tee logs/sft_phase1.log
>      Record final eval_loss as the Phase-1 baseline.
>   3. Smoke-test inference on the SFT checkpoint with one prompt.
>   4. Run Phase 1.9 RL:
>        python scripts/train.py --rl --skip-prepare 2>&1 | tee logs/rl_phase1.log
>      Record completions/mean_length and mean visual_reward as the Phase-1
>      baselines for later comparisons.
>   5. Tick 1.8 and 1.9 in plans/architecture-roadmap.md, then proceed to
>      Phase 2 (unfreeze vision LoRA) -- it's a one-line config change plus
>      retraining.
>
> Working directory is the repo root. Do not touch CLAUDE.md or rewrite the
> coord-token module. Ask before running any destructive git operations.
> ```

---

## Project at a glance

**What Elysium is**: a vision-language-action model that edits images by
predicting *drawing commands*, not pixels. The model never touches pixels
directly -- it emits structured JSON actions and a deterministic canvas
executor renders them. See `CLAUDE.md` for the rules of the road and
`README` / `src/elysium/` for the codebase layout.

**Base model**: `Qwen/Qwen3.5-4B` (VLM). LoRA adapters on language linear
layers. Vision tower currently frozen (Phase 2 will unfreeze).

**Action space**: 12 typed actions (brush, pencil, eraser, fill,
color_adjust, text_overlay, gaussian_blur, clone_stamp, scatter_brush,
pattern_brush, forward_warp, noop) with mixed continuous + categorical
parameters. Coordinates and color channels live in `[0, 255]`.

**Dataset (current)**: 219 train / 55 val sentinel-encoded chunks after the
Phase 1 prepare_data run. Down from 1361/341 in the pre-Phase-1 version
because the compress stage skips raw sessions whose `canvas_size != 256`
(many were captured at 512). A side task is to rescale those properly
(image + coords + canvas_size field) to recover the ~1.7K-chunk count;
not strictly required for Phase 1 verification.

**Training stack**: HuggingFace Transformers + TRL (SFTTrainer / GRPOTrainer)
+ PEFT (LoRA). The Unsloth FastVisionModel path is preserved for
laptop-VRAM (QLoRA + 8-bit optim) use.

---

## Architecture roadmap status

See `plans/architecture-roadmap.md` for the full plan. High-level status:

| Phase | Status | Notes |
| --- | --- | --- |
| **Phase 1** — Coord-token binning | code complete; full SFT + RL pending | sub-steps 1.1-1.7 done; 1.8 SFT + 1.9 RL are the next concrete commands |
| **Phase 2** — Unfreeze vision LoRA | not started | flip `lora.finetune_vision_layers: true` and retrain |
| **Phase 3** — Action history in prompt | not started | ~30 LoC across `format.py` + `predict.py` |
| **Phase 4** — Reward sharpening (SSIM + format bonus) | not started | ~50 LoC in `reward.py` |
| **Phase 5** — Re-evaluate gate | not started | hard go/no-go before any big architecture change |
| **Phase 6** — Optional "action expert lite" | gated | only after Phase 5 says go |
| **Phase 7** — Optional learned drawing critic | gated | only if scaling instructions past GT |

The roadmap also documents what is explicitly OUT of scope (full π0
flow-matching expert, per-session multi-turn with image history,
end-to-end episodic GRPO) until the project's scale grows ~10x.

---

## Phase 1 detail — what changed and why

The motivation: actions are dominated by continuous-valued integer fields
(coordinates in [0,255], color channels in [0,255]). Emitting these as
decimal text wastes tokens and the tokenizer can split `"128"` as
`["12","8"]` vs `["1","28"]` -- pure ambiguity noise that the model has
to learn through.

### Wire format

Each integer in a coordinate or color slot is emitted as a single dedicated
token. Three disjoint namespaces, **regular added tokens** (not "special"
-- they must survive `skip_special_tokens=True` during decode, which the
JSON stop criterion uses):

  - `<x0>..<x255>`  -- canvas column
  - `<y0>..<y255>`  -- canvas row
  - `<c0>..<c255>`  -- color channel (R/G/B/A)

Sample chunk:

```json
{"actions":[{"action_type":"brush",
             "color_rgba":[<c255>,<c0>,<c0>,<c255>],
             "stroke_size":3,"hardness":100,
             "start_point":[<x10>,<y20>],
             "end_point":[<x30>,<y40>]}]}
```

This is not strict JSON. The parser `extract_action_json` in
`src/elysium/model/action_io.py` resolves sentinels to ints before
`json.loads`. The brace-balance state machine in
`src/elysium/model/action_io.py:JsonBalanceState` is unaffected --
sentinels contain no `{`, `}`, `"`, or `\`.

### Files added or changed

  - **NEW** `src/elysium/model/coord_tokens.py`
    Owns the token namespace, `add_coord_tokens(tokenizer)` (idempotent),
    `init_coord_token_embeddings(model, tokenizer)` (seeds each new row from
    the mean embedding of the decimal-digit tokens of its integer).
  - **NEW** `src/elysium/model/loading.py`
    HuggingFace + PEFT helpers (no Unsloth): `load_base_model`,
    `load_processor`, `apply_lora`, `ensure_coord_tokens`,
    `load_adapter_for_inference`, `load_adapter_for_training`.
  - `src/elysium/schemas/actions.py` -- `ActionChunk.to_json_str`/
    `from_json_str` now emit/parse the sentinel format. System prompt
    documents the new wire format. Manual emitter avoids the
    `json.dumps`-escapes-NUL-markers issue.
  - `src/elysium/model/action_io.py` -- `extract_action_json` calls
    `resolve_tokens` before `json.loads`.
  - `src/elysium/data/format.py` -- the dataset builder now routes through
    `ActionChunk.to_json_str` so `gt_actions` and assistant messages contain
    sentinels.
  - `src/elysium/model/train.py` -- restructured to branch on
    `model.use_unsloth`. Both Unsloth (FastVisionModel +
    UnslothVisionDataCollator) and HF/PEFT (AutoModelForImageTextToText +
    PEFT LoraConfig + custom `_HFActionVisionDataCollator`) paths apply the
    coord-token surgery before LoRA wrap.
  - `src/elysium/model/rl_train.py` -- same toggle, same Phase 1 wiring.
  - `src/elysium/model/predict.py` -- same toggle.
  - `configs/train.yaml` -- new `model.use_unsloth` key, defaults to
    `false`. Other training params tuned for the bigger remote GPU.

### Tests

`tests/unit/test_coord_tokens.py` (new) covers:
  - All 768 tokens are distinct.
  - `encode_x`/`encode_y`/`encode_c` reject out-of-range.
  - `resolve_tokens` round-trips and is a no-op on plain JSON.
  - Round-trip serialization across all 12 action types.
  - Sentinel-containing chunks parse correctly via `parse_action_chunk`.
  - `JsonBalanceState` triggers exactly at the outer closing brace.

Full suite: `python -m pytest tests/ -q` -> 33 passing.

### Smoke results from the laptop (8 GB RTX 4060 Laptop)

  - 1-epoch SFT with QLoRA + Unsloth: trained to completion, checkpoint
    saved. `eval_loss = 1.427`. Note: the first-epoch *train* loss is
    visibly high (~29) because the freshly added token rows start with
    bad logit calibration; eval after the epoch is much more sensible.
    Expect both to drop within a few epochs.
  - Inference smoke on the same machine crashed with a cuDNN
    `CUDNN_STATUS_INTERNAL_ERROR` at the vision tower's Conv3d patch
    embed. Cause: cuDNN workspace OOM in tight VRAM, compounded by
    Unsloth defaulting the vision tower to 512x512 even though our
    canvases are 256x256. This is *not* a model-quality issue; it's an
    environment issue and should not reproduce on the remote machine.

---

## Files of interest (read first)

In rough priority order:

  - `CLAUDE.md` -- code rules.
  - `plans/architecture-roadmap.md` -- the phased plan.
  - `plans/HANDOFF.md` -- this document.
  - `configs/train.yaml` -- single source of training config; pay attention
    to `model.use_unsloth` and `model.load_in_4bit`.
  - `src/elysium/schemas/actions.py` -- action types, system prompt, the
    sentinel serializer.
  - `src/elysium/model/coord_tokens.py` -- coord-token namespace, embed init.
  - `src/elysium/model/loading.py` -- HF/PEFT loading helpers.
  - `src/elysium/model/train.py` -- SFT trainer with both paths.
  - `src/elysium/model/rl_train.py` -- GRPO trainer with both paths.
  - `src/elysium/model/predict.py` -- inference loop with both paths.
  - `src/elysium/model/action_io.py` -- chat-template + parse helpers.

---

## First-day playbook on the remote machine

```bash
# 0. Clone / pull and enter the repo
cd elysium
git checkout training-upgrade

# 1. Install (after PyTorch with CUDA is already installed in the env).
pip install -r requirements.txt && pip install -e .

# 2. Sanity-check environment
nvidia-smi
python -m pytest tests/ -q                # expect 33 passed
python -c "from elysium.model.loading import load_base_model; print('loading OK')"

# 3. Verify the dataset has Phase 1 sentinel tokens (already prepped, but
#    if data/processed is empty re-run prepare_data first).
python -c "from datasets import load_from_disk; d=load_from_disk('data/processed'); print(d['train'][0]['gt_actions'][:200])"
# Expect to see <x...>, <y...>, <c...> tokens.

# 4. (Optional) Tighten the config for your specific GPU. Defaults assume
#    >= 24 GB VRAM. If you have less, drop batch_size and bump
#    gradient_accumulation_steps to keep effective batch ~16.

# 5. Phase 1.8: full SFT
mkdir -p logs
python scripts/train.py --skip-prepare 2>&1 | tee logs/sft_phase1.log

# 6. Smoke inference on the SFT checkpoint
SMOKE_IMG=$(ls data/raw/frames/*/0.jpg | head -1)
python scripts/infer.py "$SMOKE_IMG" "draw a smiley face" \
    --checkpoint models/checkpoints/final \
    --output /tmp/sft_phase1.jpg

# 7. Phase 1.9: RL on top of SFT
python scripts/train.py --rl --skip-prepare 2>&1 | tee logs/rl_phase1.log

# 8. Tick the boxes in plans/architecture-roadmap.md (1.8, 1.9), commit.
```

After Phase 1 baselines are recorded, Phase 2 is a config flip:

```bash
# Phase 2: unfreeze vision LoRA
sed -i 's/finetune_vision_layers: false/finetune_vision_layers: true/' configs/train.yaml
python scripts/train.py --skip-prepare 2>&1 | tee logs/sft_phase2.log
python scripts/train.py --rl --skip-prepare 2>&1 | tee logs/rl_phase2.log
```

Compare eval_loss / mean visual_reward against Phase 1 logs. Flat or better
= keep. Regression = revert and document.

---

## Known issues & gotchas

  1. **Local-laptop trl import error** (only on Spartak's laptop):
     `cannot import name 'TRANSFORMERS_CACHE' from 'transformers.utils.hub'`
     when `from trl import GRPOConfig, GRPOTrainer` runs. Caused by
     transformers 5.5.0 dropping `TRANSFORMERS_CACHE` while `llm_blender`
     still imports it. A fresh environment install on the remote should
     pull compatible versions and avoid this. If it bites you, pin
     `transformers==4.46.*` or remove `llm_blender` from the `trl[judges]`
     extras.

  2. **Dataset shrinkage to 219/55 chunks**. The compress stage rejects
     sessions where `canvas_size != 256`. Many raw sessions are 512.
     Options:
       (a) Run the existing `scripts/rescale_images.py` AND also scale
           every coordinate in the session JSONs by 0.5 AND update the
           `canvas_size` field. The rescale_images.py alone does not
           cover (b)+(c).
       (b) Or change `CANVAS_SIZE = 256` to 512 in
           `src/elysium/schemas/actions.py` -- but the coord-token
           namespace is hard-coded at 256 (`N_VALUES`) and would need to
           double, plus all existing checkpoints would need to be
           re-trained. Not recommended.
     219 chunks is enough to verify Phase 1 mechanics; it is NOT enough
     for high-quality outputs. Address before judging Phase 2-4 results.

  3. **Unsloth's forced 512x512 image size**. The laptop run showed
     `Model does not have a default image size - using 512` and
     `Unsloth: Making 'model.base_model.model.model.visual' require gradients`.
     The HF/PEFT path (`use_unsloth: false`) avoids both. If you ever
     re-enable Unsloth, expect 4x more vision tokens than necessary --
     consider patching the pixel budget by writing to
     `processor.image_processor.size` directly before training, since
     `apply_image_pixel_budget` is a no-op on current Unsloth.

  4. **First-epoch train-loss spike**. Adding 768 new tokens with
     digit-mean init makes their initial logits poor; per-step CE during
     training inflates to ~12-30 until the LM head recalibrates. eval_loss
     post-epoch is the trustworthy signal. If train loss is still > 10
     after epoch 5, that's a real problem -- investigate the embedding init
     path (input vs output, tied/untied) in `coord_tokens.py`.

  5. **Coord-token resize must happen before LoRA wrap.** Both paths in
     `train.py` already do this. If you ever refactor, preserve the order:
     `load -> ensure_coord_tokens -> apply_lora`.

---

## Conversation summary (what the user asked, what was answered)

This handoff was generated at the end of a session that walked through:

  1. **"How does the SFT/RL training work today?"** -- Each chunk is its
     own *independent single-turn conversation*. The model never sees prior
     chunks in its context. Per-step prompt = system + user(image+
     instruction); assistant = JSON chunk. Both SFT and RL operate
     per-chunk; no multi-turn history. Temporal dependence is implicit
     through the rendered canvas image only. See
     `src/elysium/data/format.py:_to_conversation`,
     `src/elysium/model/train.py:ElysiumActionVisionDataCollator`, and
     `src/elysium/model/rl_train.py:_build_grpo_dataset`.

  2. **"How does the visual reward work? Shouldn't it be free from GT?"** --
     The current reward (`src/elysium/model/reward.py:visual_reward`) is
     *coverage-weighted pixel-imitation* against the canvas produced by
     executing the GT actions. It IS GT-conditioned; the "visual" framing
     is about renderer-equivalence (different action sequences that
     render to the same canvas get similar reward), not GT-freeness. The
     docstring at `src/elysium/model/rl_train.py:4` calling it "pure
     visual (SSIM) reward" is stale -- implementation uses coverage x MAE.
     A truly GT-free reward (Phase 7 / drawing critic) is gated by Phase 5.

  3. **"Architecture review -- do we need an action expert? per-session
     conversations? GT-free reward?"** -- Detailed analysis: no full action
     expert yet (12 typed heterogeneous actions + ~1.7K dataset don't
     justify it); per-chunk conversations are correct for compute but
     should add a short rolling history of past chunks as *text* (Phase 3);
     the reward should evolve in stages (Phase 4 sharpens GT-conditioned;
     Phase 7 adds a GT-free critic, but only if scaling). The recommended
     ranked priority is: (1) coord-token binning, (2) unfreeze vision
     LoRA, (3) action history in prompt, (4) sharpen reward, (5)
     re-evaluate -- and ONLY THEN consider Phase 6/7. Documented in
     `plans/architecture-roadmap.md`.

  4. **"Make a plan.md with checkpoints"** -- Resulted in
     `plans/architecture-roadmap.md` with checkbox-trackable sub-steps,
     acceptance criteria, and explicit out-of-scope items.

  5. **"Implement Phase 1"** -- Done. All sub-steps 1.1-1.7 ticked.
     1.8 SFT and 1.9 RL pending (this remote machine's job).

  6. **"Switch to QLoRA"** (laptop OOM at resize) -> later **"switch to
     LoRA and disable Unsloth for this run"** -- Resulted in the
     `model.use_unsloth: false` toggle, the new HF/PEFT path in
     `loading.py`, and parallel paths in `train.py` / `rl_train.py` /
     `predict.py`. Unsloth path preserved for laptop reuse.

  7. **"Export this conversation with a default prompt for remote Claude"**
     -- This document.

---

## Final notes for Claude on the remote machine

  - Don't re-implement Phase 1 -- it's already done. Just run the SFT/RL.
  - Don't rewrite `coord_tokens.py` or `loading.py`. They are tested.
  - Don't remove the Unsloth code branch. It is the laptop fallback.
  - Don't bypass `--skip-prepare` unless you intentionally want to
    regenerate `data/processed`. The current data already encodes
    sentinels.
  - After Phase 1.8 + 1.9 succeed, tick the boxes in
    `plans/architecture-roadmap.md` and start Phase 2.

Good luck.
