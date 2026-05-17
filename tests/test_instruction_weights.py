"""Phase 5.5.2: tests for per-instruction sample weighting in build_dataset."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from elysium.data.format import _load_instructions, build_dataset


def _write_instructions(path: Path) -> None:
    config = {
        "tasks": {
            "happy_task": {
                "instruction": "do happy thing",
                "sessions": ["sess_happy_1"],
            },
            "sad_task": {
                "instruction": "do sad thing",
                "sessions": ["sess_sad_1"],
            },
            "neutral_task": {
                "instruction": "do neutral thing",
                "sessions": ["sess_neutral_1"],
            },
        }
    }
    path.write_text(yaml.safe_dump(config))


def _write_chunk_file(path: Path, session: str, n_chunks: int, img_path: Path) -> None:
    # Give each chunk a distinct observation_frame and a distinct action to
    # ensure (image, action_text) uniquely identifies it — the leakage test
    # below needs this to distinguish "same content in two splits because of
    # replication" from "two unrelated chunks happening to look identical."
    chunks = []
    for i in range(n_chunks):
        per_chunk_frame = img_path.parent / f"{session}_frame_{i}.png"
        per_chunk_frame.write_bytes(b"\x89PNG\r\n\x1a\n")
        chunks.append({
            "chunk_index": i,
            "observation_frame": str(per_chunk_frame),
            "actions": [
                {
                    "action_type": "brush",
                    "color_rgba": [0, 0, 0, 255],
                    "stroke_size": 5,
                    "hardness": 100,
                    "start_point": [10 + i, 10],
                    "end_point": [20, 20 + i],
                },
                {"action_type": "noop"},
            ],
        })
    path.write_text(json.dumps({"session": session, "chunks": chunks}))


class LoadInstructionsTests(unittest.TestCase):
    def test_returns_both_text_and_name_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            instr_path = Path(tmp) / "instructions.yaml"
            _write_instructions(instr_path)
            text_map, name_map = _load_instructions(instr_path)
            self.assertEqual(text_map["sess_happy_1"], "do happy thing")
            self.assertEqual(name_map["sess_happy_1"], "happy_task")
            self.assertEqual(name_map["sess_sad_1"], "sad_task")


class BuildDatasetWeightingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)
        # Create a stand-in PNG file path (build_dataset only stores the
        # path string; no actual image read happens).
        self.img_path = self.tmp / "frame.png"
        self.img_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.chunks_dir = self.tmp / "chunks"
        self.chunks_dir.mkdir()
        self.instr_path = self.tmp / "instructions.yaml"
        _write_instructions(self.instr_path)
        for sess in ("sess_happy_1", "sess_sad_1", "sess_neutral_1"):
            _write_chunk_file(
                self.chunks_dir / f"{sess}.json", sess, n_chunks=3, img_path=self.img_path
            )

    def tearDown(self) -> None:
        self.tmp_obj.cleanup()

    def _run(self, weights, scale=4):
        out = self.tmp / "processed"
        build_dataset(
            self.chunks_dir,
            self.instr_path,
            out,
            horizon=2,
            instruction_weights=weights,
            weight_scale=scale,
        )
        from datasets import load_from_disk
        return load_from_disk(str(out))

    def test_no_weights_yields_unreplicated_records(self) -> None:
        # weight_scale collapses to 1 when no explicit weights -> 9 records
        # (3 sessions * 3 chunks * 1 copy).
        ds = self._run(weights=None)
        total = len(ds["train"]) + len(ds["validation"])
        self.assertEqual(total, 9)

    def test_uniform_weights_with_scale4_replicates(self) -> None:
        # Explicit uniform weights still take the scaled path: 3*3*4 = 36.
        ds = self._run(weights={"happy_task": 1.0, "sad_task": 1.0, "neutral_task": 1.0}, scale=4)
        total = len(ds["train"]) + len(ds["validation"])
        self.assertEqual(total, 36)

    def test_up_and_down_weight_balances_correctly(self) -> None:
        # happy: 1.5 -> 6 copies; sad: 0.5 -> 2 copies; neutral default 1.0 -> 4
        # Each session has 3 chunks: happy=18, sad=6, neutral=12 -> 36 records.
        ds = self._run(weights={"happy_task": 1.5, "sad_task": 0.5}, scale=4)
        total = len(ds["train"]) + len(ds["validation"])
        self.assertEqual(total, 36)
        # Count happy vs sad records by instruction text content.
        all_records = list(ds["train"]) + list(ds["validation"])
        happy_count = sum(
            1 for r in all_records
            if "do happy thing" in r["messages"][1]["content"][1]["text"]
        )
        sad_count = sum(
            1 for r in all_records
            if "do sad thing" in r["messages"][1]["content"][1]["text"]
        )
        neutral_count = sum(
            1 for r in all_records
            if "do neutral thing" in r["messages"][1]["content"][1]["text"]
        )
        self.assertEqual(happy_count, 18)
        self.assertEqual(sad_count, 6)
        self.assertEqual(neutral_count, 12)

    def test_zero_weight_drops_task_entirely(self) -> None:
        ds = self._run(weights={"sad_task": 0.0}, scale=4)
        all_records = list(ds["train"]) + list(ds["validation"])
        sad_count = sum(
            1 for r in all_records
            if "do sad thing" in r["messages"][1]["content"][1]["text"]
        )
        self.assertEqual(sad_count, 0)

    def test_replicated_records_never_leak_across_split(self) -> None:
        # Regression test for the §5.5.2 leakage bug: when we replicate
        # records BEFORE splitting, the same literal record can land in
        # both train and val, which makes eval_loss measure memorization
        # instead of generalization. Fix splits first, replicates within
        # each split. Verify by checking that the set of distinct
        # observation_frame paths in train and val are disjoint per task.
        ds = self._run(weights={"happy_task": 2.0, "sad_task": 2.0}, scale=4)
        # Each base record uniquely identified by (image, action_text).
        def _key(r):
            return (
                r["image"],
                r["messages"][2]["content"][0]["text"],  # assistant action JSON
            )
        train_keys = {_key(r) for r in ds["train"]}
        val_keys = {_key(r) for r in ds["validation"]}
        overlap = train_keys & val_keys
        self.assertEqual(overlap, set(),
                         f"Records leak across train/val: {len(overlap)} shared keys")


if __name__ == "__main__":
    unittest.main()
