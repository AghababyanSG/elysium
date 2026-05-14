from __future__ import annotations

import json
from itertools import takewhile
from typing import Any

from elysium.log import logger
from elysium.model.coord_tokens import resolve_tokens
from elysium.schemas.actions import ActionChunk, NoopAction, build_system_prompt, parse_action

__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "action_conversation_messages",
    "apply_action_chat_template",
    "split_prompt_completion_texts",
    "build_generation_processor_inputs",
    "strip_redacted_thinking_lead",
    "JsonBalanceState",
    "json_balance_advance",
    "extract_action_json",
    "parse_action_chunk",
]


CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def _render_user_text(instruction: str, history_actions: list[str]) -> str:
    """Format the user text segment with optional action history.

    Mirrors the training-time rendering in ``elysium.data.format._render_user_text``
    so inference prompts match exactly the format the policy was trained on.
    """
    if not history_actions:
        return instruction
    history_block = "\n".join(history_actions)
    return f"Recent actions:\n{history_block}\n\nInstruction: {instruction}"


def action_conversation_messages(
    instruction: str,
    horizon: int,
    history_actions: list[str] | None = None,
) -> list[dict[str, Any]]:
    user_text = _render_user_text(instruction, history_actions or [])
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": build_system_prompt(horizon)}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def apply_action_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        **CHAT_TEMPLATE_KWARGS,
    )


def split_prompt_completion_texts(
    processor: Any,
    prompt_messages: list[dict[str, Any]],
    completion_messages: list[dict[str, Any]],
) -> tuple[str, str]:
    p_txt = apply_action_chat_template(
        processor,
        prompt_messages,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    pc_txt = apply_action_chat_template(
        processor,
        prompt_messages + completion_messages,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    p_prefix = "".join(x for x, _ in takewhile(lambda ab: ab[0] == ab[1], zip(p_txt, pc_txt)))
    c_txt = pc_txt[len(p_prefix) :]
    return p_prefix, c_txt


def build_generation_processor_inputs(
    processor: Any,
    canvas_pil: Any,
    instruction: str,
    horizon: int,
    history_actions: list[str] | None = None,
) -> dict[str, Any]:
    messages = action_conversation_messages(
        instruction, horizon, history_actions=history_actions
    )
    text = apply_action_chat_template(
        processor,
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    return processor(text=[text], images=[canvas_pil], return_tensors="pt", padding=False)


def strip_redacted_thinking_lead(raw: str) -> str:
    t = raw.strip()
    open_t = "<think>"
    close_t = "</think>"
    if open_t in t and close_t in t:
        end = t.find(close_t)
        if end != -1:
            t = t[end + len(close_t) :].lstrip()
    return t.strip()


class JsonBalanceState:
    """Mutable state for incremental JSON-object balance tracking."""

    __slots__ = ("start", "depth", "in_str", "esc")

    def __init__(self) -> None:
        self.start: int | None = None
        self.depth: int = 0
        self.in_str: bool = False
        self.esc: bool = False


def json_balance_advance(state: JsonBalanceState, ch: str, pos: int) -> bool:
    """Advance the balance state machine by one character.

    Args:
        state: Mutable balance state.
        ch: Next character from the stream.
        pos: Absolute position of `ch` in the original string (used to record start).

    Returns:
        True when the outermost `{...}` has just been closed (depth returns to 0).
    """
    if state.start is None:
        if ch == "{":
            state.start = pos
            state.depth = 1
        return False
    if state.in_str:
        if state.esc:
            state.esc = False
        elif ch == "\\":
            state.esc = True
        elif ch == '"':
            state.in_str = False
        return False
    if ch == '"':
        state.in_str = True
    elif ch == "{":
        state.depth += 1
    elif ch == "}":
        state.depth -= 1
        if state.depth == 0:
            return True
    return False


def _first_balanced_json_object(s: str) -> str | None:
    state = JsonBalanceState()
    for i, ch in enumerate(s):
        if json_balance_advance(state, ch, i):
            assert state.start is not None
            return s[state.start : i + 1]
    return None


def extract_action_json(raw: str) -> str:
    """Return the first balanced ``{...}`` object as strict JSON text.

    Coord/color sentinels (``<xN>``/``<yN>``/``<cN>``) are resolved to their
    integer values here so downstream callers can use ``json.loads`` directly.
    """
    t = strip_redacted_thinking_lead(raw)
    blob = _first_balanced_json_object(t)
    if blob is None:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    blob = resolve_tokens(blob)
    data = json.loads(blob)
    if not isinstance(data, dict) or "actions" not in data:
        raise ValueError(f"JSON must be an object with 'actions': {blob!r}")
    return blob


def parse_action_chunk(raw_output: str, horizon: int) -> ActionChunk:
    """Parse a model-generated chunk, skipping individual malformed actions.

    Each action is validated independently; if one fails (e.g. junk in a point
    like ``[446,32dr]``), that action is dropped with a warning and the rest
    are kept. Raises ``ValueError`` if every action fails validation or if the
    ``actions`` list is empty -- silently returning a noop chunk would let the
    RL reward function score a malformed generation as a clean completion.
    """
    blob = extract_action_json(raw_output)
    raw = json.loads(blob)
    raw_actions = raw.get("actions") or []
    parsed = []
    for i, item in enumerate(raw_actions):
        try:
            parsed.append(parse_action(item))
        except (ValueError, TypeError, AssertionError) as exc:
            logger.warning(
                "Skipping invalid action {} ({}): {}", i, type(exc).__name__, exc
            )
    if not parsed:
        raise ValueError(
            f"All {len(raw_actions)} actions in chunk failed validation (or list was empty)"
        )
    return ActionChunk(actions=parsed, horizon=horizon)
