"""Coordinate-token binning: single-token representation of 0..255 integers.

Replaces decimal text for canvas coordinates and color channels with
dedicated added tokens, so the model emits one token per value instead of
1–3 digit tokens with ambiguous splits:

  <xN>, <yN>  -- canvas coordinates (x = column, y = row), N in [0, 255]
  <cN>        -- color channel (R/G/B/A), N in [0, 255]

Wire format: JSON with bare sentinels at coord/color positions, e.g.

  {"actions":[{"action_type":"brush",
               "color_rgba":[<c255>,<c0>,<c0>,<c255>],
               "stroke_size":3,"hardness":100,
               "start_point":[<x10>,<y20>],
               "end_point":[<x30>,<y40>]}]}

This is not strict JSON — sentinels are bare identifiers, not numbers. The
brace-balance extractor still works (sentinels contain no braces/quotes),
and :func:`resolve_tokens` substitutes each sentinel for its integer value
before ``json.loads``.

We use ``tokenizer.add_tokens(..., special_tokens=False)`` rather than
``add_special_tokens`` so the sentinels survive ``skip_special_tokens=True``
in decode (which the JSON stop criterion relies on).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "N_VALUES",
    "X_PREFIX",
    "Y_PREFIX",
    "C_PREFIX",
    "all_coord_tokens",
    "coord_token_ids",
    "encode_x",
    "encode_y",
    "encode_c",
    "resolve_tokens",
    "add_coord_tokens",
    "init_coord_token_embeddings",
]


N_VALUES = 256
X_PREFIX = "x"
Y_PREFIX = "y"
C_PREFIX = "c"

_TOKEN_RE = re.compile(r"<([xyc])(\d+)>")


def all_coord_tokens() -> list[str]:
    """Return every coord/color sentinel string (length 3 * N_VALUES)."""
    return (
        [f"<{X_PREFIX}{i}>" for i in range(N_VALUES)]
        + [f"<{Y_PREFIX}{i}>" for i in range(N_VALUES)]
        + [f"<{C_PREFIX}{i}>" for i in range(N_VALUES)]
    )


def coord_token_ids(tokenizer: Any) -> list[int]:
    """Return token IDs of every coord/color sentinel for the given tokenizer.

    Use this to mark the new vocabulary rows as trainable via PEFT's
    ``LoraConfig.trainable_token_indices`` — without it the rows stay frozen
    at their digit-mean init and the model cannot learn to prefer them over
    the existing decimal-digit tokens in argmax decoding.
    """
    ids: list[int] = []
    for tok in all_coord_tokens():
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert isinstance(tid, int) and tid >= 0, (
            f"tokenizer has no id for {tok!r} -- call add_coord_tokens first"
        )
        ids.append(tid)
    return ids


def encode_x(n: int) -> str:
    assert 0 <= n < N_VALUES, f"x-coord {n} out of [0, {N_VALUES})"
    return f"<{X_PREFIX}{n}>"


def encode_y(n: int) -> str:
    assert 0 <= n < N_VALUES, f"y-coord {n} out of [0, {N_VALUES})"
    return f"<{Y_PREFIX}{n}>"


def encode_c(n: int) -> str:
    assert 0 <= n < N_VALUES, f"color channel {n} out of [0, {N_VALUES})"
    return f"<{C_PREFIX}{n}>"


def resolve_tokens(text: str) -> str:
    """Replace every ``<xN>``/``<yN>``/``<cN>`` sentinel with its integer text.

    This turns a JSON-with-sentinels blob into strict JSON parseable by
    ``json.loads``. The axis prefix is intentionally discarded — the schema
    already knows which key each value belongs to.
    """
    return _TOKEN_RE.sub(lambda m: m.group(2), text)


def add_coord_tokens(tokenizer: Any) -> int:
    """Add coord/color sentinels to the tokenizer in-place.

    Idempotent: returns the number of *newly added* tokens (0 if all were
    already present, e.g. when loading a checkpoint that included them).

    Caller must invoke ``model.resize_token_embeddings(len(tokenizer))``
    afterwards, **before** wrapping the model with PEFT/LoRA — otherwise the
    LoRA-targeted LM head won't pick up the new rows.
    """
    return int(tokenizer.add_tokens(all_coord_tokens(), special_tokens=False))


def init_coord_token_embeddings(model: Any, tokenizer: Any) -> int:
    """Initialize freshly-added coord-token embedding rows from digit-token means.

    For each sentinel ``<xN>`` (and y/c variants), seed its input and output
    embedding to the mean of the embeddings of the tokens that the base
    tokenizer produces for ``str(N)``. This places the new row in roughly
    the same region of embedding space as the spelled-out number, which is
    a much better starting point than random init for early SFT.

    Returns the number of rows initialized.
    """
    import torch

    tokens = all_coord_tokens()
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    input_weight = input_emb.weight.data
    tied = (
        output_emb is None
        or output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
    )
    # Skip writing to a quantized LM head (e.g. bitsandbytes 4-bit Linear) --
    # its ``weight.data`` is a packed uint8 buffer and direct row-indexing
    # would corrupt it. In typical Unsloth QLoRA configs lm_head stays in
    # fp16/bf16, so this branch only trips on aggressive quantization setups.
    output_weight = None
    if not tied and output_emb.weight.is_floating_point():
        output_weight = output_emb.weight.data

    initialized = 0
    with torch.no_grad():
        for tok in tokens:
            m = _TOKEN_RE.match(tok)
            assert m is not None, f"unexpected token format: {tok!r}"
            n = int(m.group(2))
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            assert isinstance(tok_id, int) and tok_id >= 0, (
                f"tokenizer has no id for {tok!r} -- did you call add_coord_tokens "
                "and resize_token_embeddings first?"
            )
            digit_ids = tokenizer.encode(str(n), add_special_tokens=False)
            assert digit_ids, f"could not encode digits for {tok}: str(n)={n}"
            digit_ids_t = torch.as_tensor(digit_ids, device=input_weight.device)
            mean_vec = input_weight.index_select(0, digit_ids_t).mean(dim=0)
            input_weight[tok_id] = mean_vec.to(input_weight.dtype)
            if output_weight is not None:
                out_mean = (
                    output_weight.index_select(0, digit_ids_t).mean(dim=0)
                )
                output_weight[tok_id] = out_mean.to(output_weight.dtype)
            initialized += 1
    return initialized
