from __future__ import annotations

from typing import Any

import torch
from transformers import StoppingCriteria

from elysium.model.action_io import JsonBalanceState, json_balance_advance

__all__ = ["JsonBalanceStoppingCriteria"]


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class JsonBalanceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._state = JsonBalanceState()
        self._last_full = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        suffix = input_ids[0, self._prompt_len :]
        if suffix.shape[0] == 0:
            return False
        full = self._tokenizer.decode(suffix, skip_special_tokens=True)
        lcp = _lcp_len(self._last_full, full)
        if lcp < len(self._last_full):
            self._state = JsonBalanceState()
            to_scan = full
        else:
            to_scan = full[lcp:]
        self._last_full = full
        for ch in to_scan:
            if json_balance_advance(self._state, ch, 0):
                return True
        return False
