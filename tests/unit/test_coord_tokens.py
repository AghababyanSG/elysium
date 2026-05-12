from __future__ import annotations

import unittest

from elysium.model.action_io import JsonBalanceState, json_balance_advance, parse_action_chunk
from elysium.model.coord_tokens import (
    N_VALUES,
    all_coord_tokens,
    encode_c,
    encode_x,
    encode_y,
    resolve_tokens,
)
from elysium.schemas.actions import (
    ActionChunk,
    BrushAction,
    EraserAction,
    FillAction,
    NoopAction,
    PencilAction,
)


class TestCoordTokenStrings(unittest.TestCase):
    def test_namespace_sizes_are_disjoint(self) -> None:
        tokens = all_coord_tokens()
        self.assertEqual(len(tokens), 3 * N_VALUES)
        self.assertEqual(len(set(tokens)), len(tokens))

    def test_encode_round_trip(self) -> None:
        for n in (0, 1, 127, 128, 255):
            self.assertEqual(encode_x(n), f"<x{n}>")
            self.assertEqual(encode_y(n), f"<y{n}>")
            self.assertEqual(encode_c(n), f"<c{n}>")

    def test_encode_rejects_out_of_range(self) -> None:
        with self.assertRaises(AssertionError):
            encode_x(-1)
        with self.assertRaises(AssertionError):
            encode_y(N_VALUES)
        with self.assertRaises(AssertionError):
            encode_c(300)


class TestResolveTokens(unittest.TestCase):
    def test_resolves_all_axes(self) -> None:
        s = '{"start_point":[<x100>,<y200>],"color_rgba":[<c255>,<c0>,<c0>,<c255>]}'
        out = resolve_tokens(s)
        self.assertEqual(
            out, '{"start_point":[100,200],"color_rgba":[255,0,0,255]}'
        )

    def test_noop_on_plain_json(self) -> None:
        plain = '{"actions":[{"action_type":"noop"}]}'
        self.assertEqual(resolve_tokens(plain), plain)


class TestChunkSerializationRoundTrip(unittest.TestCase):
    def _round_trip(self, chunk: ActionChunk) -> ActionChunk:
        raw = chunk.to_json_str()
        # Sentinels must be present whenever the action carries coords/colors.
        if any(a.action_type not in {"noop", "gaussian_blur"} for a in chunk.actions):
            self.assertRegex(raw, r"<[xyc]\d+>")
        # Parser must accept the sentinel-augmented JSON.
        return ActionChunk.from_json_str(raw, horizon=chunk.horizon)

    def test_brush_round_trip(self) -> None:
        chunk = ActionChunk(
            actions=[
                BrushAction(
                    action_type="brush",
                    color_rgba=(255, 0, 0, 255),
                    stroke_size=3,
                    start_point=(10, 20),
                    end_point=(30, 40),
                    hardness=80,
                )
            ],
            horizon=1,
        )
        out = self._round_trip(chunk)
        a = out.actions[0]
        self.assertEqual(a.action_type, "brush")
        self.assertEqual(a.color_rgba, (255, 0, 0, 255))
        self.assertEqual(a.start_point, (10, 20))
        self.assertEqual(a.end_point, (30, 40))
        self.assertEqual(a.stroke_size, 3)
        self.assertEqual(a.hardness, 80)

    def test_pencil_eraser_fill_round_trip(self) -> None:
        chunk = ActionChunk(
            actions=[
                PencilAction(
                    action_type="pencil",
                    color_rgba=(0, 255, 0, 200),
                    start_point=(5, 5),
                    end_point=(250, 250),
                ),
                EraserAction(
                    action_type="eraser",
                    stroke_size=8,
                    start_point=(0, 0),
                    end_point=(100, 100),
                ),
                FillAction(
                    action_type="fill",
                    color_rgba=(0, 0, 255, 255),
                    position=(128, 128),
                ),
            ],
            horizon=3,
        )
        out = self._round_trip(chunk)
        self.assertEqual(out.actions[0].color_rgba, (0, 255, 0, 200))
        self.assertEqual(out.actions[1].start_point, (0, 0))
        self.assertEqual(out.actions[2].position, (128, 128))

    def test_noop_round_trip_has_no_sentinels(self) -> None:
        chunk = ActionChunk(
            actions=[NoopAction(action_type="noop"), NoopAction(action_type="noop")],
            horizon=2,
        )
        raw = chunk.to_json_str()
        self.assertNotRegex(raw, r"<[xyc]\d+>")
        out = ActionChunk.from_json_str(raw, horizon=2)
        self.assertTrue(out.is_terminal)


class TestParseActionChunkWithSentinels(unittest.TestCase):
    def test_parse_full_sentinel_chunk(self) -> None:
        raw = (
            '{"actions":[{"action_type":"brush",'
            '"color_rgba":[<c255>,<c0>,<c0>,<c255>],'
            '"stroke_size":3,"hardness":100,'
            '"start_point":[<x10>,<y20>],"end_point":[<x30>,<y40>]}]}'
        )
        chunk = parse_action_chunk(raw, 1)
        self.assertEqual(chunk.actions[0].start_point, (10, 20))
        self.assertEqual(chunk.actions[0].color_rgba, (255, 0, 0, 255))

    def test_mixed_sentinel_and_plain_int(self) -> None:
        # Plain ints in fields that *aren't* coord/color (stroke_size, hardness)
        # must still parse alongside sentinels.
        raw = (
            '{"actions":[{"action_type":"pencil",'
            '"color_rgba":[<c10>,<c20>,<c30>,<c255>],'
            '"start_point":[<x0>,<y0>],"end_point":[<x100>,<y100>]}]}'
        )
        chunk = parse_action_chunk(raw, 1)
        self.assertEqual(chunk.actions[0].color_rgba, (10, 20, 30, 255))


class TestStopCriterionStateMachineWithSentinels(unittest.TestCase):
    """Brace tracking must close exactly at the outer ``}`` even when the
    text contains sentinel tokens between commas."""

    def test_brace_balance_closes_after_outer_brace(self) -> None:
        chunk = ActionChunk(
            actions=[
                BrushAction(
                    action_type="brush",
                    color_rgba=(255, 0, 0, 255),
                    stroke_size=3,
                    start_point=(10, 20),
                    end_point=(30, 40),
                )
            ],
            horizon=1,
        )
        raw = chunk.to_json_str()
        state = JsonBalanceState()
        triggered_at = None
        for i, ch in enumerate(raw):
            if json_balance_advance(state, ch, i):
                triggered_at = i
                break
        self.assertIsNotNone(triggered_at)
        self.assertEqual(raw[triggered_at], "}")
        # And it triggered exactly at the *last* closing brace, not earlier.
        self.assertEqual(triggered_at, len(raw) - 1)


if __name__ == "__main__":
    unittest.main()
