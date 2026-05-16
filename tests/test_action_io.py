from __future__ import annotations

import unittest

from elysium.model.action_io import (
    apply_action_chat_template,
    extract_action_json,
    ActionParseError,
    parse_action_chunk,
    split_prompt_completion_texts,
    strip_redacted_thinking_lead,
)


class MockProcessor:
    def apply_chat_template(
        self,
        messages: list,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **kwargs: object,
    ) -> str:
        assert kwargs.get("enable_thinking") is False
        n = len(messages)
        if n == 2 and add_generation_prompt:
            return "PA"
        if n == 3 and not add_generation_prompt:
            return "PAB"
        raise AssertionError(f"unexpected call n={n} add_gen={add_generation_prompt}")


class TestApplyActionChatTemplate(unittest.TestCase):
    def test_passes_enable_thinking_false(self) -> None:
        calls: list[bool] = []

        class P:
            def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
                calls.append(kwargs.get("enable_thinking") is False)
                return "ok"

        out = apply_action_chat_template(P(), [], add_generation_prompt=True)
        self.assertEqual(out, "ok")
        self.assertTrue(calls and calls[0])


class TestSplitPromptCompletion(unittest.TestCase):
    def test_prefix_strip(self) -> None:
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": "s"}]},
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
        ]
        completion = [{"role": "assistant", "content": [{"type": "text", "text": "{}"}]}]
        p, c = split_prompt_completion_texts(MockProcessor(), prompt, completion)
        self.assertEqual(p, "PA")
        self.assertEqual(c, "B")


class TestStripThinking(unittest.TestCase):
    def test_strips_closed_block(self) -> None:
        raw = "<think>\nnoise\n</think>\n\n{\"actions\":[]}"
        self.assertTrue(strip_redacted_thinking_lead(raw).startswith("{"))


class TestExtractActionJson(unittest.TestCase):
    def test_extracts_first_object_with_actions(self) -> None:
        blob = extract_action_json('hello {"actions":[{"action_type":"noop"}]} tail')
        self.assertIn("actions", blob)

    def test_rejects_without_actions_key(self) -> None:
        with self.assertRaises(ValueError):
            extract_action_json('{"foo":1}')

    def test_rejects_plain_prose(self) -> None:
        with self.assertRaises(ValueError):
            extract_action_json("The user wants me to draw")


class TestParseActionChunk(unittest.TestCase):
    def test_parses_five_noops(self) -> None:
        raw = '{"actions":[' + ",".join(['{"action_type":"noop"}'] * 5) + "]}"
        chunk = parse_action_chunk(raw, 5)
        self.assertEqual(len(chunk.actions), 5)
        self.assertTrue(all(a.action_type == "noop" for a in chunk.actions))

    def test_skips_action_with_invalid_point(self) -> None:
        good = '{"action_type":"noop"}'
        bad = (
            '{"action_type":"brush","color_rgba":[0,0,0,255],"stroke_size":5,'
            '"hardness":100,"start_point":[10,"bad"],"end_point":[20,20]}'
        )
        raw = '{"actions":[' + good + "," + bad + "," + good + "]}"
        chunk = parse_action_chunk(raw, 5)
        self.assertEqual(len(chunk.actions), 2)
        self.assertTrue(all(a.action_type == "noop" for a in chunk.actions))

    def test_all_invalid_actions_raises(self) -> None:
        bad = (
            '{"action_type":"brush","color_rgba":[0,0,0,255],"stroke_size":5,'
            '"hardness":100,"start_point":[10,"bad"],"end_point":[20,20]}'
        )
        raw = '{"actions":[' + bad + "," + bad + "]}"
        with self.assertRaises(ValueError):
            parse_action_chunk(raw, 5)

    def test_empty_actions_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_action_chunk('{"actions":[]}', 5)

    def test_all_invalid_actions_raises_typed_error(self) -> None:
        # Phase 5.4.1: the all-fail raise is now ActionParseError so the
        # inference loop can distinguish it from a model-emitted terminal
        # noop chunk (which previously triggered false-termination via the
        # noop_chunk fallback). Subclass of ValueError for back-compat with
        # callers that still catch ValueError.
        bad = (
            '{"action_type":"brush","color_rgba":[0,0,0,255],"stroke_size":5,'
            '"hardness":100,"start_point":[10,"bad"],"end_point":[20,20]}'
        )
        raw = '{"actions":[' + bad + "," + bad + "]}"
        with self.assertRaises(ActionParseError) as cm:
            parse_action_chunk(raw, 5)
        self.assertIsInstance(cm.exception, ValueError)

        with self.assertRaises(ActionParseError):
            parse_action_chunk('{"actions":[]}', 5)

    def test_drops_action_with_sentinel_infiltration_in_scalar(self) -> None:
        # Models occasionally fuse a stray sentinel into a numeric scalar:
        # `"size":1<y90>`. resolve_tokens would flatten <y90> → 90 producing
        # `"size":190`, which then either silently passes with the wrong value
        # or fails range validation (size cap 50) and the action gets dropped
        # without explanation — costing the Predictor its termination signal.
        # The fix: detect the infiltration before resolve_tokens runs and
        # reject the action explicitly.
        good = '{"action_type":"noop"}'
        bad = (
            '{"action_type":"clone_stamp","source":[<x10>,<y20>],'
            '"destination":[<x30>,<y40>],"size":1<y90>}'
        )
        raw = '{"actions":[' + good + "," + bad + "," + good + "]}"
        chunk = parse_action_chunk(raw, 5)
        self.assertEqual(len(chunk.actions), 2)
        self.assertTrue(all(a.action_type == "noop" for a in chunk.actions))

    def test_accepts_legitimate_sentinel_placement(self) -> None:
        # Regression guard: a fully sentinel-encoded clone_stamp with sentinels
        # only inside the point/color arrays must NOT be flagged as infiltrated.
        raw = (
            '{"actions":[{"action_type":"clone_stamp",'
            '"source":[<x10>,<y20>],"destination":[<x30>,<y40>],"size":12}]}'
        )
        chunk = parse_action_chunk(raw, 1)
        self.assertEqual(len(chunk.actions), 1)
        self.assertEqual(chunk.actions[0].action_type, "clone_stamp")

    def test_drops_action_with_sentinel_in_color_scalar(self) -> None:
        # Same failure shape on a color channel: `"thickness":2<c5>` would
        # become 25 after resolve_tokens — silently shifting param values.
        good = '{"action_type":"noop"}'
        bad = (
            '{"action_type":"text_overlay","text":"hi","position":[<x10>,<y20>],'
            '"color_rgba":[<c0>,<c0>,<c0>,<c255>],"thickness":2<c5>}'
        )
        raw = '{"actions":[' + good + "," + bad + "]}"
        chunk = parse_action_chunk(raw, 5)
        self.assertEqual(len(chunk.actions), 1)
        self.assertEqual(chunk.actions[0].action_type, "noop")


if __name__ == "__main__":
    unittest.main()
