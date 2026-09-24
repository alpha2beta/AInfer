#!/usr/bin/env python3
"""Test suite for P0 Improvements:
  - I1.1: Stop sequence parsing and matching (streaming + non-streaming)
  - I1.2: Native FIM (Fill-In-The-Middle) formatting and auto-stop tokens
"""

import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "http"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))

import tok as tokenizer_tool
from server_258v import (
    EOS_TOKEN_IDS,
    FIM_STOP_TOKENS,
    FIM_STOP_TOKEN_IDS,
    IncrementalDecoder,
    StreamStopBuffer,
)


class TestStreamStopBuffer(unittest.TestCase):
    def test_empty_stop_sequences(self):
        buf = StreamStopBuffer([])
        self.assertEqual(buf.append("hello "), "hello ")
        self.assertEqual(buf.append("world"), "world")
        self.assertEqual(buf.flush(), "")
        self.assertFalse(buf.stopped)

    def test_single_char_stop(self):
        buf = StreamStopBuffer(["\n"])
        emitted = []
        for t in ["def ", "foo():", "\n", "    return 1"]:
            chunk = buf.append(t)
            if chunk:
                emitted.append(chunk)
            if buf.stopped:
                break
        self.assertEqual("".join(emitted), "def foo():")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "\n")

    def test_stop_inside_single_token(self):
        buf = StreamStopBuffer(["\n"])
        chunk = buf.append("result = 42\nnext_line")
        self.assertEqual(chunk, "result = 42")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "\n")

    def test_multi_char_stop_across_tokens(self):
        buf = StreamStopBuffer(["\n\n"])
        emitted = []
        for t in ["line 1\n", "\nline 2"]:
            chunk = buf.append(t)
            if chunk:
                emitted.append(chunk)
            if buf.stopped:
                break
        self.assertEqual("".join(emitted), "line 1")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "\n\n")

    def test_partial_prefix_not_followed_by_remainder(self):
        buf = StreamStopBuffer(["\n\n"])
        emitted = []
        for t in ["line 1\n", "still line 1"]:
            chunk = buf.append(t)
            if chunk:
                emitted.append(chunk)
            if buf.stopped:
                break
        emitted.append(buf.flush())
        self.assertEqual("".join(emitted), "line 1\nstill line 1")
        self.assertFalse(buf.stopped)

    def test_code_block_delimiter(self):
        buf = StreamStopBuffer(["```"])
        emitted = []
        for t in ["var x = 1;\n", "``", "`\nmore"]:
            chunk = buf.append(t)
            if chunk:
                emitted.append(chunk)
            if buf.stopped:
                break
        self.assertEqual("".join(emitted), "var x = 1;\n")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "```")

    def test_earliest_stop_wins(self):
        buf = StreamStopBuffer(["LONG_STOP", "SHORT"])
        emitted = []
        chunk = buf.append("prefix SHORT and LONG_STOP")
        if chunk:
            emitted.append(chunk)
        self.assertEqual("".join(emitted), "prefix ")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "SHORT")

    def test_stop_at_start_of_generation(self):
        buf = StreamStopBuffer(["\n"])
        chunk = buf.append("\n")
        self.assertEqual(chunk, "")
        self.assertTrue(buf.stopped)
        self.assertEqual(buf.matched_stop, "\n")


class TestIncrementalDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer_tool.load()

    def test_ascii_and_multibyte_utf8(self):
        text = "Hello 世界 😊 def add(a, b):\n    return a + b"
        ids = tokenizer_tool.encode(self.tok, text)
        dec = IncrementalDecoder(self.tok)
        parts = []
        for i in ids:
            p = dec.step(i)
            if p:
                parts.append(p)
        f = dec.flush()
        if f:
            parts.append(f)
        self.assertEqual("".join(parts), text)

    def test_special_tokens_preservation(self):
        text = "<|fim_prefix|>test<|fim_suffix|>end<|fim_middle|>"
        ids = tokenizer_tool.encode(self.tok, text)
        dec = IncrementalDecoder(self.tok)
        parts = []
        for i in ids:
            p = dec.step(i)
            if p:
                parts.append(p)
        f = dec.flush()
        if f:
            parts.append(f)
        self.assertEqual("".join(parts), text)


class TestFIMFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer_tool.load()

    def test_fim_wrapping(self):
        prefix = "def calculate_area(width, height):\n    "
        suffix = "\n    return area"
        wrapped = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        ids = tokenizer_tool.encode(self.tok, wrapped)
        # Check that FIM tokens are present
        self.assertIn(248060, ids)  # <|fim_prefix|>
        self.assertIn(248062, ids)  # <|fim_suffix|>
        self.assertIn(248061, ids)  # <|fim_middle|>
        decoded = tokenizer_tool.decode(self.tok, ids, skip_special_tokens=False)
        self.assertEqual(decoded, wrapped)

    def test_fim_constants(self):
        self.assertIn("<|fim_middle|>", FIM_STOP_TOKENS)
        self.assertIn("<|fim_suffix|>", FIM_STOP_TOKENS)
        self.assertIn("<|fim_prefix|>", FIM_STOP_TOKENS)
        self.assertIn("<|file_sep|>", FIM_STOP_TOKENS)
        self.assertIn(248061, FIM_STOP_TOKEN_IDS)
        self.assertIn(248065, FIM_STOP_TOKEN_IDS)


if __name__ == "__main__":
    unittest.main()
