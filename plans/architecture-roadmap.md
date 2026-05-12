# Architecture Upgrade Roadmap

Prioritized plan for changes to the Elysium VLA, ranked by ROI vs effort.
Tick each box once the acceptance criterion holds. Do not skip ahead — items
1–5 are gates for items 6–7, which only become worth doing after measuring
results from the cheaper wins.

---

## Phase 1 — Coordinate-token binning (highest-ROI single change)

Replace integer-as-text coordinates (and optionally small int params) with
dedicated special tokens `<x0>..<x255>` / `<y0>..<y255>`. Wins: ~3x shorter
completions, no digit-tokenization ambiguity (`"128"` → `["12","8"]` vs
`["1","28"]` etc.), still trivially JSON-parseable.

Budget: ~4–8 hours of code + one SFT run. Realistically half a day of
coding with a chance of one debug cycle pushing to a full day.

- [x] **1.1 Add special tokens to the tokenizer**
  - Add `<x0>..<x255>` and `<y0>..<y255>` (or a shared `<n0>..<n255>` namespace).
  - Call `tokenizer.add_special_tokens(...)` then `model.resize_token_embeddings(len(tokenizer))`.
  - **Gotcha:** resize *before* wrapping with `FastVisionModel.get_peft_model`,
    or the LM head's LoRA target won't pick up the new rows.
  - Acceptance: `tokenizer.encode("<x123>")` returns a single token id; the
    embedding/lm-head matrix has the new rows.

- [x] **1.2 Rewrite serialization in `src/elysium/schemas/actions.py`**
  - Update `ActionChunk.to_json_str` and `parse_action` so points emit/parse
    `[<x100>,<y200>]` instead of `[100,200]`.
  - Decide explicitly whether to bin: `color_rgba` channels, `stroke_size`,
    `hardness`, `radius`, `size`, `density`, etc. Recommendation: bin only
    coordinates and color channels in this pass; leave other small ints as text.
  - Acceptance: round-trip `ActionChunk.from_json_str(chunk.to_json_str())`
    is identity for every action type.

- [x] **1.3 Re-run `scripts/prepare_data.py`**
  - The new encoding must land in `data/processed/`'s `gt_actions` strings
    and the assistant text inside `messages`.
  - Acceptance: `load_from_disk` on `data/processed/` shows the new tokens
    in the assistant messages, and `len(d['train'])` matches the previous
    run (1361).

- [x] **1.4 Update `JsonBalanceStoppingCriteria`**
  - `src/elysium/model/stop_on_json.py` is character-based. Verify it still
    detects the closing `}` correctly when neighbouring tokens decode to
    multi-character strings like `<x100>`.
  - Acceptance: a unit test that feeds a tokenized assistant turn
    one-token-at-a-time triggers stop exactly at the outer brace close.

- [x] **1.5 Update tests**
  - `tests/test_action_io.py` and any schema/parse tests need fixtures
    regenerated for the new encoding.
  - Acceptance: `python -m pytest tests/` is green.

- [x] **1.6 Embedding initialization for the new tokens**
  - Random init hurts early SFT. Initialize each `<xN>` embedding as the
    *mean* of the embeddings of the digit-tokens forming the decimal
    string `str(N)` (or copy from `tokenizer.encode(str(N))` token average).
  - Apply the same init to the LM head row.
  - Acceptance: a single un-trained forward pass on a sample prompt
    produces non-NaN logits and the model's top-k continuations include
    the coord tokens.

- [x] **1.7 Verify forward pass + parse round-trip pre-training**
  - Generate one chunk from the un-tuned model with the new tokenizer,
    confirm `parse_action_chunk` accepts the output (or at least raises
    a recognizable error, not a tokenizer crash).
  - Acceptance: scripted smoke test passes locally.

- [ ] **1.8 Run SFT with new encoding**
  - `python scripts/train.py --epochs 12 --skip-prepare` (data already
    rebuilt in 1.3).
  - Acceptance: training loss curve looks comparable to the prior baseline
    (no divergence from the new-token init), final eval loss recorded.

- [ ] **1.9 Run RL pass and record baseline metrics**
  - `python scripts/train.py --rl --skip-prepare`.
  - Acceptance: completions/mean_length is much lower than the pre-binning
    run (proves tokens are doing their job), mean visual reward recorded
    for comparison against later phases.

---

## Phase 2 — Unfreeze vision LoRA

Base ViT was trained on natural photos; cannot perceive 256² synthetic
canvases precisely. Adding low-rank adapters lets the encoder adapt.

Budget: ~5 min of config + one SFT run.

- [ ] **2.1 Flip vision LoRA flag**
  - `configs/train.yaml`: `lora.finetune_vision_layers: true`, keep `r: 4` or
    `r: 8` for the vision tower if Unsloth exposes a per-tower rank; otherwise
    accept r=16 across the board.
  - Acceptance: trainable-param count printed at training start is higher
    than the pre-Phase-2 baseline; vision params show up in `model.print_trainable_parameters()`.

- [ ] **2.2 Re-train SFT + RL on top of Phase 1 binned tokens**
  - Acceptance: eval loss and RL mean-reward improve over Phase 1 baselines,
    or are at least flat — if they regress, revert and document.

---

## Phase 3 — Action history in prompt

Inject the last N=2–3 chunks of actions as *text* into the user message
to give the policy plan continuity. Do not inject past images.

Budget: ~30 lines across `format.py` and `predict.py`, plus retrain.

- [ ] **3.1 Modify `src/elysium/data/format.py`**
  - In `_to_conversation`, accept an optional `history_actions: list[str]`
    parameter and prepend it as a separate text segment in the user message
    (e.g. `"Recent actions:\n<chunk_-2>\n<chunk_-1>\n\nInstruction: ..."`).
  - In `build_dataset`, when iterating chunks of a session, pass the previous
    N chunks' `actions` JSON as history.
  - Acceptance: spot-check 3 sample records — the user message of chunk i
    contains the action JSON of chunks i-2 and i-1; chunk 0 has none.

- [ ] **3.2 Mirror the change in inference**
  - `src/elysium/model/predict.py` `Predictor.run`: maintain a deque of the
    last N executed chunks' JSON; pass it into `build_generation_processor_inputs`.
  - `src/elysium/model/action_io.py` `action_conversation_messages`: accept
    optional history and inject it identically to training-time format.
  - Acceptance: at inference, log the constructed user-text on step 3 and
    confirm it contains the JSON of steps 1 and 2.

- [ ] **3.3 Sanity-check prompt budget**
  - With N=3 and horizon=2 the history adds ~100–250 tokens. Confirm
    `training.max_seq_length: 768` still has headroom; bump to 1024 only if
    truncation warnings appear.
  - Acceptance: no truncation warnings during one training epoch.

- [ ] **3.4 Re-train and measure**
  - Acceptance: trajectory-continuity tasks (stickman, mustache, scribble
    sessions) show qualitatively smoother strokes in inference frame dumps.
    Record eval loss / RL reward delta vs Phase 2.

---

## Phase 4 — Reward sharpening (still GT-conditioned)

Smooth the reward landscape so RL has gradient further from the GT.

Budget: ~50 lines in `src/elysium/model/reward.py` + one RL run.

- [ ] **4.1 Swap MAE-accuracy for SSIM patch**
  - Compute the bounding box of `gt_mask`, take the patch from `predicted_canvas`
    and `gt_next_canvas`, and use SSIM (e.g. `skimage.metrics.structural_similarity`)
    instead of `1 - mean|pred - gt|`.
  - Keep coverage × accuracy − 0.1 × spurious shape.
  - Acceptance: unit test — two slightly-misregistered strokes (1–2 px
    offset) score within 0.05 of pixel-perfect, instead of dropping ~0.3
    as MAE does today.

- [ ] **4.2 Add format-valid bonus**
  - +0.05 (small, additive) for any non-terminal, parse-valid completion.
  - Acceptance: malformed completions still score −1; near-correct but
    wrong-colour completions score strictly above 0 + bonus.

- [ ] **4.3 Fix stale docstring**
  - `src/elysium/model/rl_train.py:4` says "pure visual (SSIM) reward" —
    update to describe the actual coverage-weighted-SSIM-plus-format reward.
  - Acceptance: docstring matches `reward.visual_reward` implementation.

- [ ] **4.4 Re-run RL and measure**
  - Acceptance: RL reward variance within GRPO groups is higher than Phase 3
    (more usable gradient), `frac_reward_zero_std` lower; collapse-detection
    callback does not trip.

---

## Phase 5 — Re-evaluate before doing anything bigger

Hard gate. Do not proceed to Phase 6 until Phases 1–4 are measured.

- [ ] **5.1 Generate side-by-side qualitative comparison**
  - Run inference on the same 5–10 prompts at: baseline, post-Phase-1,
    post-Phase-3, post-Phase-4. Save frame dumps to `outputs/comparison/`.
  - Acceptance: visual comparison saved; subjective notes recorded on
    which phases moved the needle.

- [ ] **5.2 Decide whether to proceed**
  - If quality is acceptable for the target instructions: STOP. Don't add
    architectural complexity for marginal gains on a 1.7K-chunk dataset.
  - If a clear residual gap remains AND you have a plan to scale the
    dataset past ~10K chunks / ~50 instructions: proceed to Phase 6.
  - Acceptance: explicit go/no-go documented in this file.

---

## Phase 6 — Optional: "Action expert lite" (two-head decoupling)

Only do this after Phase 5 says go. Replaces text-encoded action params
with a typed parameter head per action class.

Budget: 1–2 weeks of engineering. Significant risk.

- [ ] **6.1 Design typed parameter heads**
  - One small MLP head per action class, taking the VLM's hidden state at
    the position immediately after the `action_type` token and producing
    that class's fixed parameter vector (continuous coords, color, sizes).
  - Categorical fields (`shape`, `font_name`) stay as discrete classification
    sub-heads.

- [ ] **6.2 Training loss**
  - Token cross-entropy on `action_type` + per-head MSE on continuous params
    + per-head cross-entropy on categorical params. Loss-mask everything
    outside the assistant turn.

- [ ] **6.3 Decoder loop change**
  - Inference emits one `action_type` token per action, then the head reads
    the hidden state and produces the parameter vector — no AR decoding of
    parameter tokens.

- [ ] **6.4 SFT and RL re-train**
  - Acceptance: parse failures drop to ~0% (no more JSON brittleness),
    coordinate precision improves measurably vs Phase-1 binned tokens.

---

## Phase 7 — Optional: Learned drawing critic / GT-free reward

Do this only if scaling to instructions with no GT.

Budget: ~1 day to prototype.

- [ ] **7.1 Build positive/negative training set**
  - Positives: every observation frame from the annotation sessions.
  - Negatives: same frames noised, permuted, or replaced with model
    failure-mode outputs.

- [ ] **7.2 Train SigLIP-MLP critic**
  - Frozen SigLIP image encoder + 2-layer MLP head → P(canvas is plausibly
    human-drawn for instruction X).

- [ ] **7.3 Wire as auxiliary RL reward**
  - Additive bonus to `visual_reward`; weight small (start 0.1) so it does
    not dominate the imitation signal.
  - Acceptance: RL with auxiliary reward on a small held-out set of unseen
    instructions produces non-trivial canvas changes (vs the noop baseline
    that pure GT-conditioned reward would fall back to).

---

## Explicitly NOT in scope

These are the right answer for an industrial VLA but wrong for a 1.7K-chunk
project. Re-evaluate only after Phases 6–7 if scale grows 10x.

- Full π0 / GR00T flow-matching action expert on the heterogeneous typed
  action space.
- Per-session multi-turn conversations with retained image history (image
  tokens explode prompt size).
- End-to-end episodic GRPO with sparse outcome reward.
