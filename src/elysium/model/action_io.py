from __future__ import annotations

import json
from itertools import takewhile
from typing import Any

from elysium.schemas.actions import ActionChunk, SYSTEM_PROMPT

__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "action_conversation_messages",
    "apply_action_chat_template",
    "split_prompt_completion_texts",
    "build_generation_processor_inputs",
    "strip_redacted_thinking_lead",
    "extract_action_json",
    "parse_action_chunk",
]


CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def action_conversation_messages(instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
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
) -> dict[str, Any]:
    messages = action_conversation_messages(instruction)
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


def _first_balanced_json_object(s: str) -> str | None:
    start: int | None = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                assert start is not None
                return s[start : i + 1]
    return None


def extract_action_json(raw: str) -> str:
    t = strip_redacted_thinking_lead(raw)
    blob = _first_balanced_json_object(t)
    if blob is None:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    data = json.loads(blob)
    if not isinstance(data, dict) or "actions" not in data:
        raise ValueError(f"JSON must be an object with 'actions': {blob!r}")
    return blob


def parse_action_chunk(raw_output: str, horizon: int) -> ActionChunk:
    return ActionChunk.from_json_str(extract_action_json(raw_output), horizon=horizon)
