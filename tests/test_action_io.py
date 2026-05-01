from __future__ import annotations

import unittest

from elysium.model.action_io import (
    apply_action_chat_template,
    extract_action_json,
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


if __name__ == "__main__":
    unittest.main()
