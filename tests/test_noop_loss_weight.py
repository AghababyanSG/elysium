"""Phase 5.5.1: tests for _noop_token_positions helper in train.py.

Uses a minimal mock tokenizer that supports `return_offsets_mapping` so the
test stays decoupled from the Qwen3.5-VL tokenizer weights cache.
"""
from __future__ import annotations

import unittest

from elysium.model.train import _noop_token_positions


class _CharTokenizer:
    """One token per character, with character-aligned offsets.

    Faithful to what `_noop_token_positions` actually needs from a tokenizer:
    a callable returning a dict with `input_ids` and `offset_mapping`.
    """

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = list(range(len(text)))
        offsets = [(i, i + 1) for i in range(len(text))]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class NoopTokenPositionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = _CharTokenizer()

    def test_no_noop_returns_empty(self) -> None:
        prompt = "PROMPT_"  # 7 chars
        full = prompt + '{"actions":[{"action_type":"brush"}]}'
        positions = _noop_token_positions(
            self.tok, full, prompt, n_prompt=7, full_seq_length=len(full)
        )
        self.assertEqual(positions, [])

    def test_single_noop_marks_all_span_tokens(self) -> None:
        prompt = "P"
        segment = '{"actions":[{"action_type":"noop"}]}'
        full = prompt + segment
        positions = _noop_token_positions(
            self.tok, full, prompt, n_prompt=1, full_seq_length=len(full)
        )
        # Span: characters '{"action_type":"noop"}' within segment.
        # The char tokenizer makes each char its own token; offsets are
        # (0,1), (1,2), ... So the marked positions correspond to the
        # noop span char indices, offset by n_prompt.
        start_in_seg = segment.find('{"action_type":"noop"}')
        end_in_seg = start_in_seg + len('{"action_type":"noop"}')
        expected = list(range(1 + start_in_seg, 1 + end_in_seg))
        self.assertEqual(positions, expected)

    def test_two_noops_both_marked(self) -> None:
        prompt = ""
        segment = (
            '{"actions":['
            '{"action_type":"noop"},'
            '{"action_type":"brush"},'
            '{"action_type":"noop"}]}'
        )
        positions = _noop_token_positions(
            self.tok, segment, prompt, n_prompt=0, full_seq_length=len(segment)
        )
        s1 = segment.find('{"action_type":"noop"}')
        s2 = segment.find('{"action_type":"noop"}', s1 + 1)
        expected = (
            list(range(s1, s1 + len('{"action_type":"noop"}')))
            + list(range(s2, s2 + len('{"action_type":"noop"}')))
        )
        self.assertEqual(positions, expected)

    def test_prompt_not_prefix_returns_empty(self) -> None:
        # Guards against misuse where prompt_text was rendered slightly
        # differently from the leading slice of full_text.
        positions = _noop_token_positions(
            self.tok,
            full_text='{"actions":[{"action_type":"noop"}]}',
            prompt_text="not-a-prefix",
            n_prompt=0,
            full_seq_length=36,
        )
        self.assertEqual(positions, [])

    def test_tokenization_length_mismatch_returns_empty(self) -> None:
        # full_seq_length doesn't match the standalone segment retokenization
        # length — this simulates a boundary merge effect; we skip rather than
        # mark wrong positions.
        prompt = "P"
        segment = '{"actions":[{"action_type":"noop"}]}'
        full = prompt + segment
        # Lie about full_seq_length to force the mismatch branch.
        positions = _noop_token_positions(
            self.tok, full, prompt, n_prompt=1, full_seq_length=len(full) + 5
        )
        self.assertEqual(positions, [])


if __name__ == "__main__":
    unittest.main()
