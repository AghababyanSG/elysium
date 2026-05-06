from __future__ import annotations

import unittest

from elysium.schemas.actions import CANVAS_SIZE, parse_action


class TestActionCoordClamp(unittest.TestCase):
    def test_parse_action_clamps_segment_endpoints_to_canvas(self) -> None:
        a = parse_action(
            {
                "action_type": "pencil",
                "color_rgba": [0, 0, 0, 255],
                "start_point": [0, 0],
                "end_point": [CANVAS_SIZE, -1],
            }
        )
        self.assertEqual(a.start_point, (0, 0))
        self.assertEqual(a.end_point, (CANVAS_SIZE - 1, 0))

    def test_parse_action_legacy_trajectory_collapses_to_endpoints(self) -> None:
        a = parse_action(
            {
                "action_type": "pencil",
                "color_rgba": [0, 0, 0, 255],
                "trajectory": [[0, 0], [50, 50], [CANVAS_SIZE, CANVAS_SIZE]],
            }
        )
        self.assertEqual(a.start_point, (0, 0))
        self.assertEqual(a.end_point, (CANVAS_SIZE - 1, CANVAS_SIZE - 1))

    def test_parse_action_clamps_fill_position(self) -> None:
        a = parse_action(
            {
                "action_type": "fill",
                "color_rgba": [255, 0, 0, 255],
                "position": [CANVAS_SIZE + 10, -5],
            }
        )
        self.assertEqual(a.position, (CANVAS_SIZE - 1, 0))

    def test_parse_action_clamps_clone_stamp_source_destination(self) -> None:
        a = parse_action(
            {
                "action_type": "clone_stamp",
                "source": [-1, CANVAS_SIZE],
                "destination": [0, 0],
                "size": 10,
            }
        )
        self.assertEqual(a.source, (0, CANVAS_SIZE - 1))
        self.assertEqual(a.destination, (0, 0))


if __name__ == "__main__":
    unittest.main()
