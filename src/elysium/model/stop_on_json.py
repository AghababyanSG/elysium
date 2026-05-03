from __future__ import annotations

from typing import Any

import torch
from transformers import StoppingCriteria

from elysium.model.action_io import JsonBalanceState, json_balance_advance

__all__ = ["JsonBalanceStoppingCriteria"]


class JsonBalanceStoppingCriteria(StoppingCriteria):
    """Stop generation the moment the outermost JSON object is balanced.

    Decodes only newly emitted tokens on each call to avoid O(n^2) work.
    """

    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len
        self._state = JsonBalanceState()
        self._processed: int = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        new_ids = input_ids[0, self._prompt_len + self._processed :]
        if new_ids.shape[0] == 0:
            return False
        chunk = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        self._processed += new_ids.shape[0]
        for ch in chunk:
            if json_balance_advance(self._state, ch, 0):
                return True
        return False
