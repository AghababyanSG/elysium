# Elysium VLA — Remote Handoff (2026-05-13, updated evening)

This document is a self-contained handoff so you can resume work on a remote
GPU machine. It captures the project context, the architecture roadmap, what
was done in Phase 1, and the exact prompt to paste into Claude on the remote
machine.

**Latest update:** 2026-05-13 evening. Phase 1.8 SFT is now done (final
`eval_loss = 0.31`, token accuracy `0.925`, smoke-tested green) after a
structural fix to make the new coord-token rows trainable. Phase 1.9 RL is
still pending. See "2026-05-13 evening session — what landed" below for the
full diff.

---

## Default prompt to paste into Claude on the remote device

> ```
> I'm continuing work on the Elysium VLA project. Read these files in order
> before doing anything else:
>
>   1. plans/HANDOFF.md  -- read the "2026-05-13 evening session" section
>      first, then skim the rest for context. Phase 1.8 is now DONE; Phase
>      1.9 RL is the next concrete step.
>   2. plans/architecture-roadmap.md  -- 1.1-1.8 are ticked; 1.9 pending.
>   3. CLAUDE.md  -- the project's persistent code rules.
>
> Current state (as of the evening update):
>   - Branch: training-upgrade. NOT committed yet -- there are uncommitted
>     edits to coord_tokens.py, loading.py, train.py, and this file. Commit
>     before doing anything heavy if you want a clean rollback point.
>   - SFT checkpoint at models/checkpoints/final is the GOOD one (with
>     trainable coord-token rows). The earlier broken run is archived at
>     models/checkpoints/final_run1_broken_no_trainable_tokens for reference.
>   - 33 pytest tests pass. data/processed has sentinel-encoded actions
>     (219 train / 55 val).
>   - configs/train.yaml is preset for the remote GB10 (use_unsloth=false,
>     load_in_4bit=false). Unsloth path preserved behind use_unsloth=true.
>   - Activate the venv before running anything:
>       source .venv/bin/activate
>
> What I want next:
>   1. Sanity: nvidia-smi; python -m pytest tests/ -q  (expect 33 passed).
>   2. Smoke-test the SFT checkpoint at greedy + sample to confirm it still
>      produces parseable sentinel-encoded JSON.
>   3. Run Phase 1.9 RL:
>        python scripts/train.py --rl --skip-prepare 2>&1 | tee logs/rl_phase1.log
>      Record completions/mean_length and mean visual_reward as the Phase-1
>      baselines for later comparisons.
>   4. Tick 1.9 in plans/architecture-roadmap.md, commit, then proceed to
>      Phase 2 (unfreeze vision LoRA) -- it's a one-line config change plus
>      retraining.
>
> Working directory is the repo root. Do not touch CLAUDE.md or rewrite the
> coord-token module. Ask before running any destructive git operations.
> ```

---

## 2026-05-13 evening session — what landed

**TL;DR:** Phase 1.8 SFT is done after fixing a structural bug; Phase 1.9 RL
is the next step.

### The bug we hit

Pre-evening, the LoRA config in `src/elysium/model/loading.py:apply_lora`
used `target_modules: "all-linear"` with `modules_to_save: null` and no
`trainable_token_indices`. PEFT freezes everything except LoRA adapters on
the targeted linears. That meant the 768 newly-added coord-token rows in
`embed_tokens` (and the tied `lm_head`) **stayed at their digit-mean init
for the entire training run**, with no gradient path to update them.

Consequence: even though SFT visibly drove eval_loss from 1.51 -> 0.92 over
11 epochs, the model never actually learned to prefer the `<xN>/<yN>/<cN>`
sentinels over plain decimal-digit tokens in coord slots. Under **greedy**
decode the model emitted JSON like
`"start_point":[1,4]`, `"color_rgba":[0,0,0,5]` -- plain decimals. Under
**sample temp=1.0** it produced a chaotic mix of wrong-namespace sentinels
(`<y52><y55><y55>` stuffed into a color slot). Either way the parser
rejected the output and inference fell through to noop.

The handoff already anticipated this class of failure when it said:
> First-epoch train-loss spike. Adding 768 new tokens with digit-mean init
> makes their initial logits poor; per-step CE during training inflates to
> ~12-30 until the LM head recalibrates.
The LM head can't recalibrate if its rows are frozen.

### The fix

PEFT 0.13+ supports `LoraConfig.trainable_token_indices` -- a focused way
to make specific embedding rows learnable on top of a LoRA adapter without
unfreezing the entire `embed_tokens`/`lm_head` (which would be ~1.5B extra
params on Qwen3.5-4B). We use this for the 768 coord-token rows
(~1.97M extra trainable params).

Changes:
  - `src/elysium/model/coord_tokens.py` -- new helper `coord_token_ids(tokenizer)`
    returning the list of token IDs for the 768 sentinels. Exported in
    `__all__`.
  - `src/elysium/model/loading.py:apply_lora` -- accepts an optional
    `trainable_token_indices` kwarg and forwards it to `LoraConfig`. Default
    `None` preserves prior behaviour.
  - `src/elysium/model/train.py:_build_hf_model_and_collator` -- after
    `ensure_coord_tokens`, computes `coord_token_ids(processor.tokenizer)`
    and passes it to `apply_lora`. A comment explains why it's required.
  - `src/elysium/model/train.py` also gained `_strip_none_content`, a
    separate fix for an unrelated bug that surfaced earlier today: HF
    datasets unifies the arrow schema across all `messages[].content` items,
    so text dicts ended up with stray `image: None` keys. Qwen3.5's chat
    template checks `'image' in item` by key presence, which raised
    "System message cannot contain images" on the system text and would
    have silently rendered the user instruction text as `<|image_pad|>`.
    The helper strips None-valued keys from content dicts before
    `apply_chat_template`; both the HF and Unsloth collator paths now
    route through it.

Tied-embeddings check: Qwen3.5-4B has `tie_word_embeddings=True`
(verified at runtime). The TrainableTokens delta sits on `embed_tokens`
and flows through to the LM head automatically -- this is the property
the fix relies on. If we ever move to a model with untied heads, we'd
need to extend the delta to the LM head explicitly.

### Re-run results

Run 1 (broken, no trainable coord rows, 11 epochs to early-stop):
  eval_loss 1.51 -> 1.12 -> 1.02 -> 0.97 -> 0.94 -> 0.94 -> 0.92 -> 0.93 -> 0.92 -> 0.93 -> 0.94
  Final eval_mean_token_accuracy 0.84. Greedy smoke = decimal-digit garbage.

Run 2 (fixed, full 12 epochs):
  eval_loss 1.35 -> 0.71 -> 0.53 -> 0.46 -> 0.40 -> 0.37 -> 0.35 -> 0.34
                 -> 0.32 -> 0.32 -> 0.31 -> **0.31**
  Final eval_mean_token_accuracy **0.925**. Greedy smoke output:
    {"actions":[{"action_type":"pencil",
                 "color_rgba":[<c0>,<c0>,<c0>,<c255>],
                 "start_point":[<x109>,<y82>],
                 "end_point":[<x109>,<y84>]}, ...]}
  Parser accepts, executor runs cleanly. Sample temp=1.0 also parses.

### Files & artifacts on disk

  - `models/checkpoints/final/` -- the good run-2 SFT checkpoint. LoRA
    adapter (156 MB) + processor/tokenizer with the 768 added tokens.
  - `models/checkpoints/final_run1_broken_no_trainable_tokens/` -- archive
    of the broken run-1 checkpoint, kept for reference. Delete when
    confident.
  - `logs/sft_phase1.log` -- the good run.
  - `logs/sft_phase1_run1_broken.log` -- the broken run.
  - Working tree has uncommitted edits to: `plans/HANDOFF.md`,
    `plans/architecture-roadmap.md`, `src/elysium/model/coord_tokens.py`,
    `src/elysium/model/loading.py`, `src/elysium/model/train.py`.
    Consider committing before kicking off Phase 1.9.

### What still warrants thought

  - The dataset is still 219/55 chunks (the canvas_size=256 filter,
    documented below under "Known issues"). Phase 1.9 RL will run on this
    thin slice. If reward variance collapses (`frac_reward_zero_std`
    high), revisit the 512 -> 256 rescale to recover more sessions before
    judging Phase 2-4.
  - `load_adapter_for_inference` still emits a "resized base embeddings"
    warning at load time. That's now slightly misleading: with the fix in
    place the resize is the *base* for the digit-mean init, and PEFT
    overlays the trained delta on top. Leave it for now -- the message is
    technically still accurate ("init fallback only"), but worth tightening
    if/when you next pass through that file.
  - `min_gt_pixels: 200` in configs/train.yaml's `rl:` block was tuned to
    the previous (pre-binning) policy. With the much sharper post-binning
    policy, more rows may saturate; if RL collapses immediately, try
    bumping to 300-400 to push training onto the harder tail.

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
