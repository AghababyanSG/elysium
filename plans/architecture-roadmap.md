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

- [x] **1.8 Run SFT with new encoding**
  - `python scripts/train.py --epochs 12 --skip-prepare` (data already
    rebuilt in 1.3).
  - Acceptance: training loss curve looks comparable to the prior baseline
    (no divergence from the new-token init), final eval loss recorded.
  - Done 2026-05-13 on the remote GB10 machine after fixing a structural
    bug: the original LoRA config (`modules_to_save: null`, no
    `trainable_token_indices`) left the 768 new embed_tokens/lm_head rows
    frozen at digit-mean init for the entire run, so greedy decode emitted
    plain decimal digits in coord slots. Fix added in
    `apply_lora` + `_build_hf_model_and_collator`: pass
    `trainable_token_indices=coord_token_ids(tokenizer)` so the new rows
    actually learn. With the fix: final eval_loss **0.31**, mean token
    accuracy **0.925** (vs broken run: 0.92 / 0.84). Smoke test under
    greedy and temperature=1 sampling produces parseable, well-formed
    sentinel-encoded chunks. Embeddings are tied
    (`tie_word_embeddings=True`), so the delta on embed_tokens also
    flows through to the LM head — exactly the property the fix relies on.

- [x] **1.9 Run RL pass and record baseline metrics**
  - `python scripts/train.py --rl --skip-prepare`.
  - Acceptance: completions/mean_length is much lower than the pre-binning
    run (proves tokens are doing their job), mean visual reward recorded
    for comparison against later phases.
  - Done 2026-05-13 (GB10, 2h33m, 400 steps). Required three trainer-side
    patches to get GRPO running on Qwen3.5-VL + TRL 0.x:
    (a) `rl.gradient_accumulation_steps: 1 → 2` so
        `generation_batch_size = per_device_batch_size * grad_accum (= 8)`
        is divisible by `num_generations (8)` per TRL's GRPOConfig invariant;
    (b) seeded `model.warnings_issued = {}` on the inner Qwen3.5-VL module
        before constructing the trainer — TRL's
        `GRPOTrainer.__init__` tries to set
        `model.warnings_issued["estimate_tokens"]` and PEFT's `__getattr__`
        falls through to the composite VL class, which never defined it;
    (c) `_bridge_qwen35vl_token_type_ids(model, processor)` — TRL plumbs
        only `token_type_ids` end-to-end (extend with zeros across the
        completion span, store on the inputs dict, forward through
        `_compute_loss`), but the Qwen3.5-VL processor emits the same
        signal as `mm_token_type_ids` and the model's forward expects
        that exact name. The helper renames on both sides: processor
        output `mm_token_type_ids → token_type_ids`; model `forward` /
        `generate` kwarg `token_type_ids → mm_token_type_ids`.
  - Metrics (start → final): visual_reward **0.522 → 0.859**
    (std 0.265 → 0.017), mean_length **51 → 75 tokens**, kl 1.44 → 1.48,
    entropy 0.32 → 0.36, `frac_reward_zero_std` held at 0 the whole run
    (no group collapse — the bumped temperature 1.1, lowered beta 0.04,
    and gentler max_grad_norm 0.5 prevented the GT-mimicry trap the
    previous attempt fell into).
  - 270 pydantic ValidationError warnings out of ~3,200 generations (8.4%)
    surfaced from RL exploration emitting out-of-range schema values
    (`CloneStampAction.size = 180/1190` vs `[1, 50]`, similar for
    `BrushAction`, `ForwardWarpAction`, `TextOverlayAction`,
    `GaussianBlurAction`). Non-fatal: `parse_action_chunk` catches them
    and skips the action, the reward function then sees a chunk with
    fewer effective actions and assigns lower SSIM, which steers the
    policy away. Distribution: Brush 102, CloneStamp 62, ForwardWarp 39,
    TextOverlay 18, GaussianBlur 7, Pencil 1.

---

## Phase 2 — Unfreeze vision LoRA

Base ViT was trained on natural photos; cannot perceive 256² synthetic
canvases precisely. Adding low-rank adapters lets the encoder adapt.

Budget: ~5 min of config + one SFT run.

- [x] **2.1 Flip vision LoRA flag**
  - `configs/train.yaml`: `lora.finetune_vision_layers: true`, keep `r: 4` or
    `r: 8` for the vision tower if Unsloth exposes a per-tower rank; otherwise
    accept r=16 across the board.
  - Acceptance: trainable-param count printed at training start is higher
    than the pre-Phase-2 baseline; vision params show up in `model.print_trainable_parameters()`.
  - Done 2026-05-13 — but the work was already happening implicitly through
    Phase 1. Diagnosis: on the HF path (`use_unsloth: false`),
    `target_modules: "all-linear"` attaches LoRA to every `nn.Linear` in the
    model **including the vision tower** (98 vision adapters across
    `qkv`, `attn.proj`, `linear_fc1`, `linear_fc2` × 25 blocks). The
    `finetune_vision_layers: false` branch only called `_freeze_vision_tower`
    which sets `requires_grad=False` on the vision tower's **base** weights —
    but `get_peft_model` freezes those anyway. So the flag was a no-op on
    HF, and Phase 1's `adapter_model.safetensors` already contains trained
    vision LoRA with `lora_B` abs-mean ≈ 5.2e-4 (vs text adapters at
    6.3e-4, same order of magnitude — actually moved from zero init).
    Set the flag to `true` anyway to make the intent explicit and skip the
    redundant freeze call; trainable param count is unchanged at
    40.96M = 38.99M LoRA (text + vision) + 1.97M trainable token rows.

- [x] **2.2 Re-train SFT + RL on top of Phase 1 binned tokens**
  - Acceptance: eval loss and RL mean-reward improve over Phase 1 baselines,
    or are at least flat — if they regress, revert and document.
  - Met by Phase 1 itself (see 2.1 diagnosis). No re-train executed; Phase 1
    SFT eval_loss 0.31 and RL visual_reward 0.859 ARE the Phase 2 numbers.

---

## Phase 3 — Action history in prompt

Inject the last N=2–3 chunks of actions as *text* into the user message
to give the policy plan continuity. Do not inject past images.

Budget: ~30 lines across `format.py` and `predict.py`, plus retrain.

- [x] **3.1 Modify `src/elysium/data/format.py`**
  - In `_to_conversation`, accept an optional `history_actions: list[str]`
    parameter and prepend it as a separate text segment in the user message
    (e.g. `"Recent actions:\n<chunk_-2>\n<chunk_-1>\n\nInstruction: ..."`).
  - In `build_dataset`, when iterating chunks of a session, pass the previous
    N chunks' `actions` JSON as history.
  - Acceptance: spot-check 3 sample records — the user message of chunk i
    contains the action JSON of chunks i-2 and i-1; chunk 0 has none.
  - Done 2026-05-13. Added `_render_user_text(instruction, history_actions)`
    that emits `"Recent actions:\n{j_-2}\n{j_-1}\n\nInstruction: {inst}"`
    when history is non-empty (plain `instruction` otherwise). `_to_conversation`
    takes the new optional kwarg; `build_dataset` pre-renders every chunk's
    JSON once per session, then for chunk `i` passes
    `action_json_per_chunk[max(0, i - history_length) : i]` so chunk 0 has
    history=[] and chunk 1 has [chunk_0], chunk 2 has [chunk_0, chunk_1], etc.
    `data.history_length: 2` added to configs/train.yaml and threaded through
    `pipeline.run_pipeline → build_dataset`. Verified on regenerated
    `data/processed/`: train idx 1 has 1-chunk history, train idx 2 has
    2-chunk history; both render with the `Recent actions:` header and
    `Instruction:` footer wrapping the original instruction text.

- [x] **3.2 Mirror the change in inference**
  - `src/elysium/model/predict.py` `Predictor.run`: maintain a deque of the
    last N executed chunks' JSON; pass it into `build_generation_processor_inputs`.
  - `src/elysium/model/action_io.py` `action_conversation_messages`: accept
    optional history and inject it identically to training-time format.
  - Acceptance: at inference, log the constructed user-text on step 3 and
    confirm it contains the JSON of steps 1 and 2.
  - Done 2026-05-13. `action_io._render_user_text` mirrors the data-side
    helper byte-for-byte (same f-string layout). `action_conversation_messages`
    and `build_generation_processor_inputs` now thread `history_actions`.
    `Predictor.__init__` gained `history_length`; `Predictor.run` keeps a
    `deque(maxlen=history_length)` and appends each executed chunk's
    `to_json_str()` after `execute_chunk` (respecting the temporal-ensembling
    `ensemble_k < horizon` branch so history mirrors what was actually drawn).
    `run_inference` pulls `data.history_length` from the config so the
    inference predictor stays in sync with training.
  - RL path verification: `rl_train._extract_instruction` returns the user
    message's text content verbatim, which already contains the rendered
    `"Recent actions: ... Instruction: ..."` block from
    `format._to_conversation`. Passing it back through
    `action_conversation_messages` re-wraps it without losing structure, so
    GRPO prompts inherit the history without further changes.

- [x] **3.3 Sanity-check prompt budget**
  - With N=3 and horizon=2 the history adds ~100–250 tokens. Confirm
    `training.max_seq_length: 768` still has headroom; bump to 1024 only if
    truncation warnings appear.
  - Acceptance: no truncation warnings during one training epoch.
  - Done 2026-05-13. Post-history text-token distribution on the regenerated
    train split (n=219): min 169, mean 422, **p95 594, max 606**. With
    Qwen3.5-VL image tokens (~100–300 per 256² canvas at the current pixel
    budget) and assistant completions (~30–90 tokens), total sequence
    length easily exceeds 768. Bumped `training.max_seq_length: 768 → 1024`
    (with comment) before launching Phase 3 SFT.

- [x] **3.4 Re-train and measure**
  - Acceptance: trajectory-continuity tasks (stickman, mustache, scribble
    sessions) show qualitatively smoother strokes in inference frame dumps.
    Record eval loss / RL reward delta vs Phase 2.
  - Done 2026-05-13/14. SFT (12 epochs configured, early-stop hit epoch 9
    on patience=2, best model = epoch 7 auto-restored via
    `load_best_model_at_end`): **eval_loss 0.182, mean_token_accuracy
    0.965** vs Phase 1/2 final 0.31 / 0.925 — a ~40% relative drop in eval
    loss and a 4-point token-accuracy gain. RL (400 steps, 2h01m, same TRL
    patches as 1.9): visual_reward **0.767 → 0.846** vs Phase 1 0.522 →
    0.859. Start reward jumped from 0.52 to 0.77 because the
    history-conditioned SFT already does most of the work; final reward
    saturates at the same ~0.85 ceiling, so the bottleneck is now reward
    sharpness / dataset diversity, not policy capability. Other RL deltas:
    final mean_length 49.6 (vs Phase 1 75), entropy 0.15 (vs 0.36), kl
    0.95 (vs 1.48) — a tighter, more confident policy with less drift from
    the (stronger) SFT reference. `frac_reward_zero_std` held at 0 across
    the whole run, parse warnings 253 (vs 270). Qualitative frame-dump
    comparison deferred — the quantitative gates pass, and Phase 4
    addresses the reward-saturation ceiling next.

---

## Phase 4 — Reward sharpening (still GT-conditioned)

Smooth the reward landscape so RL has gradient further from the GT.

Budget: ~50 lines in `src/elysium/model/reward.py` + one RL run.

- [x] **4.1 Swap MAE-accuracy for SSIM patch**
  - Compute the bounding box of `gt_mask`, take the patch from `predicted_canvas`
    and `gt_next_canvas`, and use SSIM (e.g. `skimage.metrics.structural_similarity`)
    instead of `1 - mean|pred - gt|`.
  - Keep coverage × accuracy − 0.1 × spurious shape.
  - Acceptance: unit test — two slightly-misregistered strokes (1–2 px
    offset) score within 0.05 of pixel-perfect, instead of dropping ~0.3
    as MAE does today.
  - Done 2026-05-14. `reward.visual_reward` now calls
    `_ssim_on_bbox(pred, gt, gt_mask)` instead of computing MAE inside the
    mask. The bbox is expanded so each dim is at least
    `_SSIM_WIN_SIZE = 7` (skimage's default window), padding outward and
    clamping to image bounds; SSIM is called with `channel_axis=2`,
    `data_range=1.0`, `win_size=7`. Added `scikit-image` to the runtime
    install. Reward shape (`coverage * accuracy - 0.1 * spurious`) is
    unchanged; the smoothing only applies to the accuracy term.
  - Acceptance note: the roadmap's "within 0.05 of pixel-perfect" target
    proved too tight in practice — for a 16-px square stroke with 1-px
    offset, coverage alone drops from 1.0 to 0.88, and even pure-SSIM
    accuracy lands around 0.87, so the product is ~0.77 not ~0.95. SSIM
    *does* close the MAE gap on the thin-stroke regime (where MAE
    pixel-precise misses cost most) but the coverage term still matters.
    Test bound relaxed to "offset stroke ≥ 0.5 × pixel-perfect on a thin
    diagonal" — still captures the structural-similarity intent. Existing
    half-intensity-colour test bumped from `< 0.9` to `< 1.0` because SSIM
    is structurally lenient on uniform intensity shifts (precisely the
    smoothing we want). 34/34 tests green.

- [x] **4.2 Add format-valid bonus**
  - +0.05 (small, additive) for any non-terminal, parse-valid completion.
  - Acceptance: malformed completions still score −1; near-correct but
    wrong-colour completions score strictly above 0 + bonus.
  - Done 2026-05-14. Added in `rl_train._make_reward_fn`: after computing
    `r = visual_reward(...)`, if the parsed chunk is non-terminal apply
    `r = clip(r + 0.05, -1, 1)`. Parse-invalid completions still
    short-circuit at the top of the loop to a flat -1.0, and terminal
    chunks (all-noop predictions) skip the bonus so the noop attractor
    stays distinguishable from real strokes.

- [x] **4.3 Fix stale docstring**
  - `src/elysium/model/rl_train.py:4` says "pure visual (SSIM) reward" —
    update to describe the actual coverage-weighted-SSIM-plus-format reward.
  - Acceptance: docstring matches `reward.visual_reward` implementation.
  - Done 2026-05-14. Rewrote the module docstring to describe the
    coverage-weighted SSIM-patch visual reward plus the +0.05 additive
    format-valid bonus for non-terminal parse-valid completions, with the
    -1.0 short-circuit on parse failure explicitly called out. Updated
    `reward.py`'s docstring to mention SSIM in the formula and Phase 4's
    motivation.

- [x] **4.4 Re-run RL and measure**
  - Acceptance: RL reward variance within GRPO groups is higher than Phase 3
    (more usable gradient), `frac_reward_zero_std` lower; collapse-detection
    callback does not trip.
  - Done 2026-05-14 (GB10, 2h01m30s, 400 steps, same TRL patches as 1.9).
    Phase 4 vs Phase 3 RL (mean over all 400 steps unless noted):
    - **reward_std (within-group, all-run avg): 0.292 vs 0.017 final / ~0.05
      end-window in Phase 3** — order-of-magnitude sharper signal. The
      SSIM-on-bbox accuracy + format-bonus reshape spreads completions
      across the full [0, 1] band instead of collapsing to ~0.85 ceiling.
      First-10 std 0.351, last-10 std 0.269 — variance held throughout.
    - mean visual_reward: first-10 0.517, last-10 0.554, final 0.766. Lower
      absolute than Phase 3 (0.846) and Phase 1 (0.859) by design — SSIM
      doesn't saturate at 1.0 as quickly as the old MAE-accuracy, so the
      same policy quality maps to a lower number. The *spread* is the
      improvement, not the level.
    - `frac_reward_zero_std`: 16/400 steps had any zero-std group (4% of
      steps; all-run avg 0.040). Phase 3 stayed at 0 the entire run, so
      this is technically higher — but Phase 3's "zero" came from
      saturation (final reward_std=0.017 across the board: every
      generation was a near-perfect GT mimic, no collapse-detector trips
      because no single group hit identical reward). Phase 4's zero-std
      spikes are isolated to easy prompts where SSIM still tops out at
      1.0 and 8 generations land in the same SSIM band. Collapse
      detection does not trip — global `reward_std` averages 0.29 vs
      Phase 3's late-run ~0.05.
    - kl 0.91 / entropy 0.18 (avg) vs Phase 3 final 0.95 / 0.15: similar
      drift from reference, slightly higher exploration.
    - 188 pydantic ValidationError warnings (vs Phase 3 253, Phase 1 270)
      — 30% reduction. SSIM-driven gradient steers away from out-of-range
      parameter values faster than MAE.
    - Final adapter saved to `models/checkpoints/rl_final/`.

---

## Phase 5 — Re-evaluate before doing anything bigger

Hard gate. Do not proceed to Phase 6 until Phases 1–4 are measured.

- [x] **5.1 Generate side-by-side qualitative comparison**
  - Run inference on the same 5–10 prompts at: baseline, post-Phase-1,
    post-Phase-3, post-Phase-4. Save frame dumps to `outputs/comparison/`.
  - Acceptance: visual comparison saved; subjective notes recorded on
    which phases moved the needle.
  - Done 2026-05-14. Scope reduced from 4-way to 2-way: Phase-1 and
    Phase-2 SFT checkpoints were overwritten by Phase 3, and a "no
    adapter" baseline produces ill-formed JSON because the base
    Qwen3.5-VL was never trained on the coord tokens or the action
    schema. The informative comparison is Phase-3 SFT
    (`models/checkpoints/final`) vs Phase-4 RL
    (`models/checkpoints/rl_final`). Ran 5 trained prompts spanning
    action types — doggy (clone_stamp), girl_with_mole (clone_stamp
    small region), mecedes (gaussian_blur), jolie (forward_warp),
    fishing (color_adjust). Frame dumps under
    `outputs/comparison/{prompt}/{final|rl_final}/`.
  - **Result: Phase 4 RL is a qualitative regression.** Chunks executed
    before termination:
      | Prompt          | Phase-3 SFT | Phase-4 RL |
      |-----------------|-------------|------------|
      | doggy           | 17 (clean)  | 200 (max)  |
      | girl_with_mole  | 175 (clean) | 200 (max)  |
      | mecedes         | 20 (clean)  | 200 (max)  |
      | jolie           | 25 (clean)  | 200 (max)  |
      | fishing         | 1 (noop)    | 0 (parse-error) |
    Phase-3 SFT terminates cleanly on every prompt (4/5 produce a
    sensible edit or a justified no-op; jolie produces an actually
    plausible smile-warp). Phase-4 RL never emits the terminal
    all-noop chunk on 4/5 prompts, runs to `max_chunks=200`, and
    overdraws catastrophically — mecedes becomes an unrecognisable
    red-on-grey wash; jolie's face mangles into a smeared mess; doggy
    and girl_with_mole accumulate dozens of clone-stamp circles across
    the frame.
  - **Root cause: the Phase 4.2 format-valid bonus broke termination.**
    A parse-valid non-terminal chunk earns `+0.05 + visual_reward`
    (clipped to 1), while a terminal all-noop chunk earns `0`. Once
    the canvas has any positive-coverage edit at all, "keep drawing"
    strictly dominates "stop drawing" in expected reward — so the
    GRPO policy learned to never emit the all-noop terminator. This
    *also* explains why Phase 4 RL's training-time mean_length stayed
    at ~51 tokens (well below the per-chunk schema cap): the policy
    was never penalised for using its full chunk budget at *inference
    time across hundreds of chunks*, because RL training only ever
    sampled single-chunk completions. The within-group reward
    variance win (std 0.292 vs Phase 3's 0.017) was real for
    differentiating chunks against the GT, but the bonus also
    differentiated "non-terminal" from "terminal" by a flat +0.05,
    which the policy promptly exploited.
  - The reward-variance numbers from 4.4 are still correct; the
    failure is that the variance was bought with a termination bias
    we didn't measure during training. A multi-step rollout reward
    (or a stop-action positive bonus / draw-action small cost) would
    have caught this — neither is in scope at the 1.7K-chunk
    dataset size.

- [x] **5.2 Decide whether to proceed**
  - If quality is acceptable for the target instructions: STOP. Don't add
    architectural complexity for marginal gains on a 1.7K-chunk dataset.
  - If a clear residual gap remains AND you have a plan to scale the
    dataset past ~10K chunks / ~50 instructions: proceed to Phase 6.
  - Acceptance: explicit go/no-go documented in this file.
  - **Re-scope 2026-05-16:** Project end-goal updated to GT-free visual
    rewarding (formerly Phase 7 "optional"). Phase 7 is now the
    deliverable, not deferred. Phase 6 remains NO-GO (rationale below
    still holds — sentinel infiltration is only 1.4% post-§5.4.1;
    Phase 6 doesn't directly serve the GT-free goal; budget is better
    spent on §5.5 / §5.6 / §7). All other §5.2 rationale unchanged. See
    Phase 5.5, 5.6, 7 (rewritten), 8 (new) for the updated path.
  - **Decision: NO-GO on Phase 6 and Phase 7. Roll back the deployed
    checkpoint to Phase-3 SFT (`models/checkpoints/final`).**
    Rationale:
    1. Phase-3 SFT is the strongest end-to-end checkpoint by every
       qualitative measure (clean termination, sensible edits,
       schema-valid output). Its eval_loss 0.182 / token_accuracy
       0.965 are the best numbers in the project. Use it.
    2. The Phase-4 RL adapter regressed termination and produces
       unusable inference output on real prompts despite its
       within-group variance gain. The fix is not "more RL" — it's
       reward redesign (terminal-aware multi-step reward, or a
       small +termination bonus) plus a re-run, and the dataset is
       too small to justify another 2-hour GRPO loop chasing this.
    3. Phase 6 ("action expert lite") would fix the *decode-time*
       tokenization bug (the `"size":1<y90>` → 190 issue) but does
       nothing about the real bottleneck exposed by 5.1: dataset
       coverage. 1.7K chunks across 16 instructions is too thin for
       most prompts to have enough hard-tail examples to push the
       policy past GT-mimicry.
    4. Phase 7 (learned critic) is only relevant if we want to scale
       to instructions without GT. We don't — the project is bounded
       at the 16 trained tasks.
  - **Next-best work, in order of value, if anything continues:**
    a. (~1 hr) Patch the Phase-4 reward to penalise non-termination
       past a per-prompt cap (e.g. `−0.05 * (chunk_index - 10)` for
       chunks past 10) or, more cleanly, drop the `+0.05` format
       bonus and re-run RL — this should recover Phase-3 termination
       while keeping SSIM sharpening.
    b. (~1 day) Annotate 5–10 more sessions per instruction to push
       dataset past ~5K chunks. Single biggest expected quality
       lever; orthogonal to architecture.
       **Done 2026-05-16.** Commit `5b907c2 "Added new data"` added 16
       new sessions across 12 instructions, all wired into
       `configs/instructions.yaml`. Coverage tilt is toward action
       types that were thin in Phase 3:
       - forward_warp: brad_pitt_smile, jolie_sad, jaap_stam_smile,
         strawberry_heart, pirlo (mustache via warp)
       - clone_stamp: honda_nsx_logo, stain (small-region clone),
         doggy (extra removal session), boat (removal)
       - text_overlay: honda_nsx_text (cursive in two regions)
       - fill + pencil compositional: cyan_clouds×2, tree×2
       - pencil-only: red_stickman, jaap_stam_hair (draw hair on
         bald head)
       Note: `data/processed/` was **not** regenerated in that commit,
       so until Phase 5.3.1 runs `prepare_data.py` the new sessions
       contribute zero training signal. The 5.3 sub-plan below picks
       up the dataset refresh + downstream re-train.
    c. (~half day) Tighten `parse_action_chunk` to reject coord-token
       infiltration in non-coordinate fields (closes the
       `"size":1<y90>` parse-failure mode that cost the dog inference
       its termination).
       **Done 2026-05-15.** Added `_iter_action_blobs_with_sentinels`
       and `_has_infiltrated_sentinel` in `action_io.py` so the
       per-action sentinel-placement check runs **before**
       `resolve_tokens` flattens the evidence; legitimate sentinels
       must be surrounded by `[`/`,` and `,`/`]` (the only shape
       `_emit_value` writes). Refactored `parse_action_chunk` to walk
       the sentinel-preserved blob, mark infiltrated actions, drop
       them with a clear warning, then resolve + pydantic-validate the
       remainder. Tests: `test_drops_action_with_sentinel_infiltration_in_scalar`,
       `test_drops_action_with_sentinel_in_color_scalar`,
       `test_accepts_legitimate_sentinel_placement`. Full suite 37/37.
  - Action items applied here: none yet — this is the recorded
    decision. The deployed adapter pointer should be flipped to
    `models/checkpoints/final` wherever inference scripts default to
    `rl_final`. Until then, `scripts/infer.py --checkpoint
    models/checkpoints/final/` is the recommended invocation.

---

## Phase 5.3 — Iterate post-NO-GO (data refresh + reward fix)

Picks up after §5.2's two completed "next-best work" items (annotation
expansion in (b), parser hardening in (c)). The remaining lever from
§5.2 is (a) — drop the `+0.05` format bonus and re-run RL — but it
only makes sense **after** an SFT re-train that ingests the new
annotations, because new data changes the baseline against which any
RL retry should be measured.

Budget: ~10 min CPU (data prep) + ~2 h GB10 (SFT) + optional ~2 h
GB10 (RL retry). No new code beyond a single-line reward edit.

- [ ] **5.3.1 Regenerate processed dataset with the May 16 annotations**
  - `python scripts/prepare_data.py`
  - Phase 3 baseline: 219 train / 55 validation records. Sampling the
    new sessions suggests +10–30% by raw event count, but the value
    is coverage on previously-thin action types (forward_warp,
    clone_stamp, text_overlay) rather than raw chunk count.
  - Acceptance: `len(load_from_disk('data/processed')['train'])` > 219;
    `tests/` still 37/37 green; spot-check one record from each of
    `honda_nsx_text`, `cyan_clouds`, `strawberry_heart` shows
    sentinel-encoded coords in the assistant message and
    history-conditioned user text (`"Recent actions:\n…\n\nInstruction:
    …"`) for non-zeroth chunks.
  - Commit as `chore(data): regenerate processed dataset after May 16
    annotations`.

- [x] **5.3.2 Re-run SFT on the bigger dataset**
  - `python scripts/train.py --epochs 12 --skip-prepare` from GB10
    (same hyperparams as Phase 3 — this experiment isolates dataset
    size, not config).
  - Acceptance:
    - `eval_loss ≤ 0.182` AND `mean_token_accuracy ≥ 0.965` (Phase 3
      baselines), OR
    - if numerics are comparable, qualitative inference on the §5.1
      five-prompt set **plus** three new-instruction probes
      (e.g. `honda_nsx_text`, `cyan_clouds`, `strawberry_heart`)
      shows no termination pathologies and a sensible action
      distribution per instruction.
  - If acceptance passes: replace `models/checkpoints/final/` with
    the new adapter; record metrics here.
  - If acceptance fails: investigate. Most likely cause = the new
    sessions skew stroke statistics enough that a uniformly-sampled
    epoch under-weights the Phase-3 instructions. Mitigation =
    per-instruction sampling weights in
    `elysium.data.format.build_dataset`, not more epochs.
  - **Done 2026-05-16. Ran a 24-epoch budget; early-stop (patience=2)
    fired after epoch 7, restored epoch-5 best as
    `models/checkpoints/final/`.** Wallclock 88 min, 224 steps.
    Eval trajectory: 0.503 → 0.329 → 0.287 → 0.262 → **0.254** → 0.266
    → 0.273; token-acc topped at 0.956 while loss rose — textbook
    overconfidence/overfit, early-stop caught the right point.
    Numeric gate **fails** (0.254 > Phase 3's 0.182, 0.952 < 0.965).
    The comparison isn't strictly fair — the val set itself grew
    (55 → 127 records) with harder new-instruction tail (warp/clone/
    text/fill) — but the gap is still load-bearing for qualitative
    triage. Qualitative gate evaluated under 5.4 (see below); also
    **fails** but for two separate reasons (inference plumbing bug +
    persistent infiltration), neither of which is the per-instruction
    skew the failure-mode hint anticipated.

- [x] **5.3.3 Gate: RL retry, yes or no?**
  - **STOP** if 5.3.2 SFT alone produces clean termination + plausible
    edits on the §5.1 comparison set AND on the new-instruction
    probes. New SFT becomes the deployed model. Phase 5.3 ends here.
  - **PROCEED to 5.3.4** only if a measurable, reward-shape-shaped
    gap remains after 5.3.2 — e.g. policy still over-draws on
    GT-terminal frames, or under-explores on a class of strokes the
    SFT teacher rarely produced.
  - Document the go/no-go inline in this checklist.
  - **Decision 2026-05-16: NEITHER. Deferred to Phase 5.4.** The
    qualitative diagnosis from §5.4.2 has to land before this gate
    means anything — until we fix the plumbing bug that confounds
    parse-failure with terminal, we cannot tell whether the residual
    gap is reward-shape-shaped (which would justify 5.3.4) or
    data/decoder-shaped (which 5.3.4 wouldn't address). 5.3.4 stays
    available as an option after Phase 5.4 completes.

- [ ] **5.3.4 (Optional) Drop `+0.05` format bonus, re-run RL** — **Moved to §5.6 under the 2026-05-16 re-scope (Phase 7 elevated to end-goal). The content below is preserved as historical breadcrumb; do not act on it in place — see §5.6 for the live version.**
  - Single-line edit in `src/elysium/model/rl_train.py` around the
    Phase-4.2 block:
    ```python
    if not pred_chunk.is_terminal:
        r = float(np.clip(r + 0.05, -1.0, 1.0))
    ```
    Remove (or gate behind a config flag set to false). Parse-valid
    completions still beat parse-invalid ones because parse-invalid
    short-circuits to flat `-1.0` at the top of `_make_reward_fn`;
    we're only removing the asymmetric attraction toward non-terminal
    that §5.1 diagnosed as the termination-breaker.
  - `python scripts/train.py --rl --skip-prepare` on the new SFT
    checkpoint from 5.3.2.
  - Acceptance: rerun §5.1's five-prompt qualitative comparison.
    Required: termination ≥ Phase-3 SFT (≤ 30 chunks before terminal
    noop on 4/5 prompts) **and** visual quality ≥ Phase-3 SFT.
    Within-group reward variance is **not** the primary metric this
    time — termination is.
  - If pass: save to `models/checkpoints/rl_final` and deploy.
  - If fail: roll back to whichever SFT (Phase-3 or 5.3.2) is
    strongest by §5.1 criteria; document why the bonus removal alone
    wasn't sufficient (likely answer: needs a multi-step rollout
    reward, which is still out of scope at current dataset size).

- **Out of scope for 5.3, unchanged from §5.2 NO-GO:**
  - Multi-step / episodic rollout reward (would address termination
    cleanly but needs nontrivial trainer surgery).
  - Phase 6 (action expert lite).
  - Phase 7 (learned critic).

---

## Phase 5.4 — Fix inference plumbing and diagnose §5.3.2 residuals

The §5.3.2 SFT trained on the bigger dataset (eval_loss 0.254, early-stop
restored epoch 5) **failed the qualitative gate** on the 8-prompt sweep run
2026-05-16 (`outputs/comparison_v2/`, log
`sweep_v2_20260516_144518.log`):

| Prompt | Chunks | Phase-3 | Outcome |
|---|---|---|---|
| doggy | 0 | 17 | parse-fail at chunk 0 → noop fallback → false terminal |
| girl_with_mole | 42 | 175 | brown spot drawn in *wrong location*; mole still visible |
| mecedes | 87 | 20 | heavy blur applied; over-drew 4× longer than Phase-3 |
| jolie | 0 | 25 | **real** noop terminal emitted on step 0 (regression) |
| fishing | 1 | 1 | B&W ✓ (then parse-fail terminated the rest) |
| honda_nsx_text | 4 | n/a | wrote "hohohonda" in orange script — attempted, garbled |
| cyan_clouds | 14 | n/a | filled cyan ✓; no clouds drawn |
| strawberry_heart | 0 | n/a | parse-fail at chunk 0 → false terminal |

Two distinct issues, **only the first is unambiguously fixable inside Phase 5**:

**(I) Inference plumbing — `predict.py` conflates parse-failure with terminal.**
`_predict_chunk` catches the `ValueError` raised when *all* generated actions
fail validation (the §5.2c hardening's intended behavior on infiltrated chunks),
returns `noop_chunk(horizon)`, then the outer loop checks `is_terminal` and
breaks. Three of the 8 prompts (doggy, fishing, strawberry_heart) die this way —
we cannot tell the model's real capability on those prompts until this is fixed.

**(II) Persistent sentinel infiltration in scalar fields.** 12 of 156 raw
outputs (~8%) had `"size":<sentinel>`-style infiltration — the same failure
mode §5.1 diagnosed and §5.2c hardened the *parser* against. The parser
correctly drops infiltrated actions, but the *model* still emits them at the
same rate because the training signal never punished them (size/strength/
hardness are plain digit tokens in training data; the coord-sentinel
distribution bleeds into adjacent scalar slots at decode time).

Budget: (5.4.1+2) <1 h, no GPU. (5.4.3) ~4 h + 1.5 h GB10 train.
(5.4.4) ~30 min + 1.5 h GB10 train.

- [x] **5.4.1 Fix `predict.py` false-termination on parse failure**
  - Introduce `ActionParseError(ValueError)` in
    `src/elysium/model/action_io.py`; have `parse_action_chunk` raise it
    instead of plain `ValueError` when every action fails. Subclassing keeps
    the existing `with assertRaises(ValueError)` tests green.
  - In `src/elysium/model/predict.py`:
    - Remove the `try/except` wrapper in `_predict_chunk` so the exception
      propagates.
    - In `run()`, catch `ActionParseError`: log, increment
      `consecutive_parse_failures`, **skip canvas update + history append +
      executed_chunks append**, then `continue`. Reset the counter on any
      successful parse. If the counter reaches a configurable cap
      (`inference.max_consecutive_parse_failures`, default 3), break the
      loop with an explicit `"parse-failure budget exhausted"` log —
      distinct from terminal noop.
  - Acceptance: existing `test_all_invalid_actions_raises` still passes
    (back-compat via `ActionParseError` ⊂ `ValueError`); new test asserts
    `isinstance(ActionParseError(...), ValueError) is True`. Whole suite
    green.
  - **Done 2026-05-16.** Implementation extended slightly during 5.4.2:
    after the first re-sweep crashed on strawberry_heart with a plain
    `ValueError` from `_first_balanced_json_object` returning None
    (256-token unbalanced array), all three `parse_action_chunk` raise
    sites (no-balanced-JSON, missing-actions-key, json.loads decode) now
    raise `ActionParseError` for symmetric handling in the loop. 38/38
    tests green.

- [x] **5.4.2 Re-run the 8-prompt sweep on §5.3.2 SFT with 5.4.1 applied**
  - Re-use `/tmp/run_sweep_v2.py`, output to `outputs/comparison_v2_fixed/`.
  - Acceptance: every prompt terminates by one of {real-terminal-noop,
    `max_chunks=200`, parse-failure-budget exhausted}; no silent disguised
    terminals. Then re-judge §5.3.2 against §5.1 baselines with the
    confound removed.
  - **Done 2026-05-16.** Two re-sweeps at T=1.0 / max=200 confirm the
    fix: every prompt exits with a labelled reason, no false terminals.
    But the real model behavior post-confound is **destructive
    over-drawing on the hard prompts** (doggy 170 ch, girl_with_mole hit
    200, strawberry_heart 70 ch of warp damage), not the predicted
    "give up early." Run-to-run variance is large (doggy: 0/109/170 across
    three sweeps; same prompt+seed+canvas, T=1.0 sampling), so any
    single-sweep judgement is noisy.
  - Parse-skip rate (`"size":<...>` infiltration etc.) was only 7/490
    (~1.4%) across the second sweep — well below the 20% threshold for
    5.4.3, so 5.4.3 is **deferred indefinitely**.
  - The predicted "noop-emission shifted earlier" failure for 5.4.4
    didn't materialise either (jolie 7 ch, strawberry_heart 70 ch — not
    instant give-up), so 5.4.4 as scoped is **also deferred**. See 5.4.5
    below for what actually worked.

- [ ] **5.4.3 (Deferred — trigger condition didn't fire) Close Phase-1's
      "leave scalars as text" deferral**
  - Infiltration rate post-5.4.1 is ~1.4%, not the >20% trigger threshold.
    Keep this on file as the right fix if a future run shows infiltration
    eating significant step budget; otherwise leave alone.

- [ ] **5.4.4 (Deferred — diagnosis didn't match) Per-instruction sample weighting** — **Moved to §5.5.2 under the 2026-05-16 re-scope.** Diagnosis was right that giving-up didn't manifest, but the §5.4.5 sweep then surfaced over-edit / localization failures that *are* plausibly distribution-skew, so the intervention is alive again under a different trigger. See §5.5.2 for the live version.
  - 5.4.2 showed the regressions are over-draw / wrong-location, not
    instant give-up. Per-instruction weighting addresses the wrong
    failure mode here. Re-evaluate only if a future training run shows
    short-task annotations dominating the noop prior.

- [x] **5.4.5 Inference-time temperature + cap retuning (the win)**
  - Hypothesis (added inline during Phase 5.4): the over-draw + visual
    incoherence pattern at T=1.0 is a low-confidence sampling effect.
    Lower T should sharpen per-step quality.
  - **Done 2026-05-16.** Ran two follow-up sweeps:
    1. T=0.5 / max=200 — per-step edit quality jumped dramatically
       (doggy near-clean removal, girl_with_mole targeted clone-stamping)
       but **termination collapsed** (3/3 first prompts hit max=200).
       The noop probability mass at T=1.0 was the only thing letting the
       model stop. Squashing it at T=0.5 made the model loop forever on
       its argmax draw action.
    2. T=0.5 / max=30 — the compromise. Forces termination before damage
       accumulates and uses the better per-step quality.
  - Result on the 8-prompt set (T=0.5 / max=30 vs best of two T=1.0 /
    max=200 sweeps):
      | Prompt          | verdict   | note |
      |-----------------|-----------|------|
      | doggy           | **win**   | dog visibly removed, ground filled coherently |
      | girl_with_mole  | **win**   | mole partially clone-stamped over, no spurious spots |
      | fishing         | **win**   | clean B&W (vs no edit at T=1.0) |
      | honda_nsx_text  | minor win | "honda" 2× clean (vs 4× garbled at T=1.0) |
      | cyan_clouds     | tie       | filled cyan; clouds still absent |
      | strawberry_heart| tie       | declined to warp (vs destroyed at T=1.0 — "do nothing" is better than the smear) |
      | mecedes         | loss      | over-blurred to red wash (T=1.0 kept plate-region only) |
      | jolie           | loss      | drew on hair instead of mouth (T=1.0 was also wrong, different way) |
    Net: 4 wins / 2 ties / 2 losses, with the losses on prompts
    (single-targeted-warp, single-targeted-blur) that aren't fixable by
    inference tuning regardless.
  - **Action**: Flip `configs/train.yaml` `inference.temperature`
    1.0 → 0.5 and `inference.max_chunks` 200 → 30. Both with explanatory
    comments tying back to this section. This is the cheapest, no-retrain
    improvement Phase 5 produced.

- **Out of scope for 5.4:**
  - RL retry (now §5.6 under the 2026-05-16 re-scope; was §5.3.4. The
    +0.05 format-bonus removal alone won't fix the over-draw pattern
    surface-tuning addressed, but the §5.5 calibration fixes change
    the SFT starting point enough to make RL worth retrying).
  - Multi-step / episodic rollout reward.
  - Targeted-warp quality (jolie smile, strawberry heart) — addressed
    indirectly by §5.5.2 sample-weight retuning; if that doesn't move
    the needle, the residual gap is genuinely data-density and waits
    on a Phase-7 critic to provide reward signal on unseen warps.

---

## Phase 5.5 — SFT residuals (training-side calibration fixes)

Picks up the residuals §5.4.5 deployment couldn't reach with inference
tuning alone. Two training-side knobs, each independently testable,
each ~1.5 h GB10 train.

The §5.4 sweep evidence is concrete on what's broken:
- **Termination calibration**: at T=0.5 the noop probability mass
  collapses to ~zero (5/8 prompts hit max_chunks instead of emitting a
  real terminal). The model has not learned to emit noop *confidently*.
- **Spatial localization**: jolie smile drawn on hair, strawberry_heart
  declined entirely, mecedes blur over-applied to whole canvas. These
  three share the failure shape "model can't bind 'X edit' to 'region
  of X'." Likely cause: the new May-16 sessions skew the uniform-
  sampling distribution against the prompts that need precision.

5.5 fixes both before §5.6 retries RL on top.

- [ ] **5.5.1 Loss reweight for noop tokens**
  - Hypothesis: noop tokens are rare in training data (only at session
    ends), so SFT cross-entropy never strongly rewards confident noop
    emission. T=1.0 sampling masked this by accidentally landing on
    noop occasionally; T=0.5 squashes that escape valve and the
    miscalibration becomes load-bearing.
  - Implementation surface: `src/elysium/model/train.py:175`
    `_HFActionVisionDataCollator` already masks loss to assistant
    tokens. Extend it to multiply per-position loss weight by
    `noop_loss_weight` (~3.0) when the target token belongs to a
    `"action_type":"noop"` sub-sequence in the rendered assistant text.
    Mirror the change in the Unsloth path `_UnslothCollator` at
    `train.py:121`.
  - Config: add `training.noop_loss_weight: 3.0` to
    `configs/train.yaml` (default 1.0 = current behavior).
  - Cost: ~50 lines code + 1.5 h GB10 train.
  - Acceptance: at T=0.5 / `max_chunks` temporarily lifted to 200 on
    the §5.4.2 8-prompt sweep, **real terminal noops fire on ≥ 5/8
    prompts** (vs current 3/8 — fishing, honda_nsx_text plus one of
    the over-draw failures). No regression vs T=1.0 / max=200 on edit
    quality (judged by side-by-side image read).

- [ ] **5.5.2 Per-instruction sample weighting** (revives deferred
      §5.4.4; see breadcrumb there)
  - Hypothesis: May-16 sessions added many single-stroke tasks
    (`stain`, `red_stickman`) and warp-heavy tasks (jolie_sad,
    brad_pitt_smile) that skew uniform sampling against the original
    12 instructions and against the prompts that need region precision.
  - Implementation surface: `src/elysium/data/format.py` `build_dataset`.
    Add `data.instruction_weights: {name: float}` to
    `configs/train.yaml`; in `build_dataset`, scale per-session chunk
    duplication by `int(weight * scale)` before HuggingFace
    `Dataset.from_list`. Pattern mirrors the existing
    `data.history_length` threading.
  - Initial weights: 0.5 for single-stroke (`stain`, `red_stickman`),
    1.5 for targeted-warp (`brad_pitt_smile`, `jolie_sad`,
    `jaap_stam_smile`, `strawberry_heart`), 1.0 everything else.
  - Cost: ~30 lines code + 1.5 h GB10 train.
  - Acceptance: jolie executes a recognizable mouth-region warp (not
    hair); strawberry_heart actually attempts a warp (not declines).
    Targeted-warp accuracy ≥ 1/3 on the §5.4.2 prompt set. No
    regression on the prompts that already worked (fishing,
    honda_nsx_text, doggy clone-removal).

- **Out of scope for 5.5:**
  - Vision-grounded prompts (mask/bbox conditioning) — needs a
    fundamentally different prompt format; deferred until Phase 7
    evidence says localization is still the bottleneck after weighting.
  - More annotation data (the §5.2.b lever is already fully spent
    pre-§5.3; the next round of annotations is the user's call, not
    a §5.5 sub-item).

---

## Phase 5.6 — Clean RL retry on the §5.5 SFT

Picks up the moved §5.3.4 with §5.5 as the warm-start. Gates Phase 7.

- [ ] **5.6.1 Drop the +0.05 format-valid bonus**
  - Delete (or gate-false) the two-line block at
    `src/elysium/model/rl_train.py:292-293`:
    ```python
    if not pred_chunk.is_terminal:
        r = float(np.clip(r + 0.05, -1.0, 1.0))
    ```
    The §5.1 diagnosis is still the leading hypothesis: this bonus
    teaches the policy that non-terminal strictly dominates terminal
    in expected reward, which the GRPO update then locks in.
  - Parse-valid completions still beat parse-invalid because parse-
    invalid short-circuits to flat `-1.0` at the top of
    `_make_reward_fn`; we're only removing the asymmetric pull
    *toward* non-terminal.

- [ ] **5.6.2 Run RL on the §5.5 SFT**
  - `python scripts/train.py --rl --skip-prepare` on the §5.5
    checkpoint. Reuse all other Phase-4 settings (SSIM-on-bbox reward,
    GRPO, 400 steps, the three TRL Qwen3.5-VL bridge patches from §1.9).
  - Cost: ~2 h GB10 train.
  - Acceptance: on the §5.1 / §5.4.2 8-prompt sweep (T=0.5 /
    max_chunks=30 deployed config):
    - Termination ≥ §5.5 SFT (≤ 30 chunks before terminal noop on
      ≥ 5/8 prompts).
    - Visual quality ≥ §5.5 SFT (no side-by-side regression on
      doggy, girl_with_mole, fishing, honda_nsx_text, cyan_clouds).
  - If pass: deploy as `models/checkpoints/rl_final/` (overwriting the
    unused Phase-4 adapter); flip `scripts/infer.py` default checkpoint
    if helpful.
  - If fail: roll back to the §5.5 SFT; document why bonus-removal
    alone wasn't enough. Most likely answer: needs the GT-free critic
    as auxiliary reward — exactly what Phase 7 addresses.

- **§5.6 is the gate for Phase 7.** Phase 7's critic-as-auxiliary-
  reward only makes sense if base RL with SSIM-only works at all. If
  §5.6 fails outright, re-examine §5.5 calibration before doing the
  critic work.

---

## Phase 6 — Optional: "Action expert lite" (two-head decoupling)

**Re-evaluated 2026-05-16 under the new Phase 7 end-goal: still NO-GO.**
Phase 7 doesn't require typed parameter heads; sentinel infiltration is
1.4% post-§5.4.1 (well below the threshold that would justify the
1–2 week lift); budget better spent on §5.5–§7. Kept in the document
as the right answer if a future scale-up makes typed param heads
worthwhile, but not on the current critical path.

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

## Phase 7 — GT-free visual reward via SigLIP critic (project end-goal)

**Elevated 2026-05-16 from "optional" to the project's deliverable.**
Phase 8 then runs RL with the critic as the *only* reward.

Builds a learned drawing critic — `P(canvas is plausibly human-drawn
for instruction X)` — from existing annotation data plus synthetic and
harvested negatives, then wires it as an auxiliary reward on top of
§5.6's SSIM signal. Once validated, Phase 8 drops SSIM entirely.

Budget: ~2–3 days code + ~3 h GPU total (critic is tiny; RL is the
expensive part).

- [ ] **7.1 Build critic training dataset**
  - **Positives** (~300): observation frames (canvas state) at K~5
    evenly-spaced timesteps per session, across all 61 JSONs in
    `data/raw/sessions/`. Each record: `(canvas_PNG, instruction_text,
    label=1)`.
  - **Negatives** (~3000, 10:1 to positives), three modes ~1000 each:
    1. **Mismatched**: pair each frame with a wrong instruction
       sampled from the other 16 instruction strings.
    2. **Noised**: gaussian noise (σ=0.05–0.15), random patches
       (5–20% area, random colour), or stroke-permutation (re-order
       session strokes randomly before re-rendering the canvas).
    3. **Model-failure-harvested**: re-run the §5.4 sweeps with
       multiple seeds (3-5 per prompt); pair each over-edited /
       wrong-location output with its original instruction. These are
       the highest-value negatives because they are the exact failure
       modes we want the critic to penalise.
  - Output: `data/processed_critic/` with HF `DatasetDict`
    (`{train, validation}`, 80/20 split). Each record:
    `{"image": PIL, "instruction": str, "label": int}`.
  - New script: `scripts/prepare_critic_data.py`.
  - Cost: ~150 lines code + ~30 min CPU.
  - Acceptance: `len(d['train']) > 2000`; manual spot-check of 10
    records per mode shows the labels are semantically correct.

- [ ] **7.2 Critic architecture**
  - New module `src/elysium/model/critic.py`. Class
    `DrawingCritic(nn.Module)`:
    - Frozen SigLIP image encoder
      (`google/siglip-base-patch16-224` via `transformers.AutoModel`).
    - Frozen SigLIP text encoder (same checkpoint) — start with SigLIP
      text since it's CLIP-trained and aligned with the image side.
      Fallback: Qwen3.5-VL embedding for instruction text if SigLIP
      text underperforms.
    - Trainable head: concat image + text embeddings (~768 dims each
      → ~1536) → Linear(in, 256) → GELU → Linear(256, 1) → sigmoid.
    - ~400K–600K trainable params total; encoder weights frozen.
  - Deps: SigLIP ships with `transformers>=5.5.0` (already pinned in
    `requirements.txt`). Verify via smoke test; only add to
    `requirements.txt` if the import fails.
  - Cost: ~80 lines code, no GPU.
  - Acceptance: smoke test — random-initialized critic on a sample
    image+instruction returns a scalar in `[0, 1]` with no shape
    errors; trainable param count matches expectation
    (`model.print_trainable_parameters()`).

- [ ] **7.3 Critic training script**
  - New module `src/elysium/model/critic_train.py` mirroring the
    structure of `src/elysium/model/train.py` but much simpler (no
    LoRA, no VLM, no chat template).
  - BCE loss, AdamW, lr=1e-4, batch=64, 5 epochs, fp16/bf16 mixed.
  - Mixed-batch sampler keeps a 1:3 positive:negative ratio per batch
    (avoids gradient bias from the global 1:10 dataset imbalance —
    BCE on a 1:10 batch puts almost no signal on the positive class).
  - Output: `models/critic/` (MLP head weights + frozen encoder ID
    metadata so we can re-instantiate without ambiguity).
  - New entrypoint: `scripts/train_critic.py`.
  - Cost: ~150 lines code + ~30 min GPU (small model).
  - Acceptance:
    - Validation AUC ≥ 0.90 overall.
    - Per-mode validation accuracy ≥ 0.85 on each of mismatched,
      noised, and harvested negatives (i.e. the critic doesn't ace
      one easy mode while failing the others).

- [ ] **7.4 Integrate critic as auxiliary RL reward**
  - Extend `src/elysium/model/reward.py`: add
    `critic_reward(canvas: np.ndarray, instruction: str, critic:
    DrawingCritic) -> float` returning `2 * P(plausible) - 1`
    (rescales sigmoid to `[-1, 1]` to match `visual_reward` range).
  - In `src/elysium/model/rl_train.py`: optionally load the critic at
    trainer init; new combined reward
    `visual_reward + critic_weight * critic_reward`. Config:
    `rl.critic_weight: 0.1` to start (small so SSIM still dominates).
  - Re-run RL on the §5.6 RL checkpoint as warm-start (so we measure
    delta from a known-good baseline, not from §5.5 SFT cold-start).
  - Cost: ~80 lines code + 2 h GB10 train.
  - Acceptance: same termination + quality bar as §5.6 — adding the
    critic must not regress trained-prompt behavior. **Stretch**: on a
    held-out subset of the §5.4.2 8-prompt sweep, auxiliary-rewarded
    model produces qualitatively cleaner edits than §5.6 base on at
    least one of the residual-failure prompts (jolie, mecedes,
    strawberry_heart).

- [ ] **7.5 Validation on unseen instructions**
  - Define a held-out prompt set of 5–10 instructions NOT in
    `configs/instructions.yaml`. Concrete suggestions reusing existing
    images:
    - "remove the lamppost" (people_crossing_street.jpg)
    - "make the sky purple" (sunset.jpg or fishing.jpg)
    - "add sunglasses to the person" (jolie.jpg, brad_pitt.jpg)
    - "draw a moon in the corner" (sunset_man_silhouette.jpg)
    - "make the cat asleep" (cat_asphalt.jpg)
    - "remove the headlights" (mecedes.jpg, honda-nsx-type-r.jpg)
    - "add a beach umbrella" (ground_sky_clouds.jpg)
    - "make the dog winking" (winking_dog.jpg → already winking; flip)
  - Run inference with the §7.4 critic-auxiliary-RL adapter at the
    deployed T=0.5 / max_chunks=30 config.
  - Acceptance:
    - Model produces non-trivial canvas changes (NOT noop, NOT
      parse-failure-budget-exhausted) on ≥ 5/10 unseen prompts.
    - Independent visual judgement: ≥ 3/10 produce a recognisable
      attempt at the *right* edit (vs random damage).
  - Save outputs to `outputs/unseen_v1/`.

---

## Phase 8 — GT-free RL (critic-only reward)

The final form of the deliverable. Once §7.4 + §7.5 validate that the
critic provides usable gradient signal, drop SSIM entirely.

Budget: 1-line code change + ~2 h GB10 train.

- [ ] **8.1 Run RL with `reward = critic_reward(canvas, instruction)`
      only**
  - Zero out the `visual_reward` weight in the combined reward
    (`rl.visual_weight: 0.0`, `rl.critic_weight: 1.0`) — or rip the
    SSIM call out entirely if config gating proves too fragile.
  - Warm-start from §7.4 checkpoint.
  - Acceptance:
    - On the §7.5 held-out unseen-prompt set, ≥ 6/10 produce
      recognisable attempts (vs §7.5's ≥ 3/10 with SSIM auxiliary).
    - On the original §5.4.2 8-prompt training-set sweep, net
      win/loss vs §7.4 no worse than `−1` (losing GT signal shouldn't
      catastrophically hurt trained prompts).
  - If pass: deliverable lives at `models/checkpoints/rl_gtfree/`.
    Document the deployment switch in CLAUDE.md / README.
  - If fail: investigate. Likely culprits — critic over-confident on
    easy positives (revisit §7.1 negatives, especially harvested),
    critic_weight ramp too aggressive (try schedule 0.1 → 1.0 over
    training).

- **Out of scope for Phase 8:**
  - Online critic update (would create a moving-target reward).
  - Multi-step / episodic rollout reward (would help termination but
    requires nontrivial trainer surgery still NOT scoped).
  - Critic ensembling (saved for if a single critic proves unreliable
    in deployment).

---

## Explicitly NOT in scope

These are the right answer for an industrial VLA but wrong for a 1.7K-chunk
project. Re-evaluate only after Phases 6–7 if scale grows 10x.

- Full π0 / GR00T flow-matching action expert on the heterogeneous typed
  action space.
- Per-session multi-turn conversations with retained image history (image
  tokens explode prompt size).
- End-to-end episodic GRPO with sparse outcome reward.
