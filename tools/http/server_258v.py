#!/usr/bin/env python3
"""Task T8.1 - T8.3: Persistent Resident In-Process HTTP Server for Intel Arc 140V (258V).

Maintains model weights and Level Zero recorded command lists permanently
resident in unified system memory across requests (Zero model reloading, zero
command list rebuilding).

Endpoints:
  GET  /healthz              -> Runtime and device health status
  GET  /readyz               -> Readiness check
  GET  /v1/models            -> Model listing
  POST /v1/chat/completions  -> OpenAI-compatible chat completion (streaming SSE & full JSON;
                                Hermes-native function calling: `tools` are rendered by the
                                pinned template and <tool_call> XML is returned as `tool_calls`)
  POST /v1/completions       -> OpenAI-compatible prompt completion

Features:
  - T8.1: Single-process in-memory persistent daemon (libainfer_258v.so via ctypes)
  - T8.2: Single-flight execution with bounded wait queue, client disconnect cancellation, and timeouts
  - T8.3: /healthz and /readyz with Level Zero device health monitoring (zeDeviceGetStatus)
"""

import argparse
import ctypes
import json
import os
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as tokenizer_tool

SO_PATH = os.path.join(REPO_ROOT, "tools", "decode", "libainfer_258v.so")
SYSROOT_LIB = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
SPV_PATH = os.path.join(REPO_ROOT, "tools", "kernels_258v", "all_kernels.spv")

MODEL_ID = "tiel-coder-35b"
MODEL_ALIAS = "symrex/Tiel-Coder-35B-A3B-Genesis-Hermes"

# Ensure Level Zero loader finds driver
os.environ["LD_LIBRARY_PATH"] = SYSROOT_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")


# T8.4: native Hermes tool-calling bridge (OpenAI-compatible).
#
# The pinned chat template renders an OpenAI-style `tools` list into a
# <tools> system block and instructs the model to emit
#   <tool_call><function=NAME><parameter=P>value</parameter>...</function></tool_call>
# These helpers convert between that wire format and the OpenAI JSON shape
# that agents (e.g. OpenCode) expect.
_TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>)?",
    re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_START_RE = re.compile(r"<tool_call>|<function=")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Structural markers that skip_special_tokens=True used to strip. I4.3 keeps
# skip=False in JSON mode so <think> tags survive for the split; these are
# stripped structurally (trailing-only) instead.
_TRAILING_MARKERS = ("<|endoftext|>", "<|im_end|>", "<|fim_middle|>",
                     "<|fim_suffix|>", "<|fim_prefix|>", "<|fim_pad|>", "<|file_sep|>")


def _strip_trailing_markers(text):
    """Remove structural special-token strings from the end of generated text."""
    s = text or ""
    while True:
        t = s.rstrip()
        for m in _TRAILING_MARKERS:
            if t.endswith(m):
                s = t[:-len(m)]
                break
        else:
            return s


def _resolve_enable_thinking(body, is_chat):
    """I4.3: resolve native-reasoning opt-in from the request body.

    Accepts OpenAI-style `reasoning_effort` ("none" disables) and
    Anthropic-style `thinking` (bool or {"type": "enabled"/"disabled"}).
    Default is ON for chat (this is a reasoning model; the pinned template
    opens `<think>` unless explicitly closed) and OFF for plain completions.
    """
    if not is_chat:
        return False
    enabled = True
    t = body.get("thinking", None)
    if isinstance(t, dict):
        enabled = str(t.get("type", "enabled")).lower() not in ("disabled", "none", "false", "no", "0")
    elif isinstance(t, str):
        enabled = t.lower() not in ("disabled", "none", "false", "no", "0", "")
    elif t is not None:
        enabled = bool(t)
    re_ = body.get("reasoning_effort", None)
    if isinstance(re_, str) and re_.lower() in ("none", "disabled", "false", "no"):
        enabled = False
    return enabled


class ThinkSplitter:
    """Incremental <think>/</think> splitter for native reasoning (I4.3).

    When enabled, generation starts inside an open thinking block (the
    prompt ends with `<think>`), so text routes to `reasoning` until the
    first `</think>`, then to `content`. A short holdback absorbs tags
    split across token boundaries; a model-echoed reopen `<think>` at the
    start is consumed. When disabled, everything passes through as content
    (zero behavior change vs the pre-I4.3 path).
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.in_think = bool(enabled)
        self.started = not bool(enabled)
        self.buf = ""

    def _close_holdback(self):
        hold = 0
        maxk = min(len(self.buf), len(_THINK_CLOSE) - 1)
        for k in range(maxk, 0, -1):
            if _THINK_CLOSE.startswith(self.buf[-k:]):
                hold = k
                break
        return hold

    def feed(self, text):
        """Consume new text; returns a list of (kind, segment) with
        kind in ("reasoning", "content")."""
        if not text:
            return []
        if not self.enabled:
            return [("content", text)]
        self.buf += text
        out = []
        while True:
            if not self.started:
                stripped = self.buf.lstrip()
                if stripped.startswith(_THINK_OPEN):
                    # Model echoed the prompt's open tag: keep any leading
                    # whitespace as reasoning, consume the tag.
                    ws = self.buf[:len(self.buf) - len(stripped)]
                    if ws:
                        out.append(("reasoning", ws))
                    self.buf = stripped[len(_THINK_OPEN):]
                    self.started = True
                    continue
                if stripped == "" or _THINK_OPEN.startswith(stripped):
                    return out  # too short to decide; wait for more text
                self.started = True
                continue
            if self.in_think:
                idx = self.buf.find(_THINK_CLOSE)
                if idx != -1:
                    if idx > 0:
                        out.append(("reasoning", self.buf[:idx]))
                    self.buf = self.buf[idx + len(_THINK_CLOSE):]
                    self.in_think = False
                    if self.buf.startswith("\n"):
                        self.buf = self.buf[1:]
                    continue
                hold = self._close_holdback()
                emit_up_to = len(self.buf) - hold
                if emit_up_to > 0:
                    out.append(("reasoning", self.buf[:emit_up_to]))
                    self.buf = self.buf[emit_up_to:]
                return out
            if self.buf:
                out.append(("content", self.buf))
                self.buf = ""
            return out

    def flush(self):
        """Emit any residual buffered text in the current state."""
        if not self.buf:
            return []
        seg = self.buf
        self.buf = ""
        if not self.enabled or not self.in_think:
            return [("content", seg)]
        return [("reasoning", seg)]


def _split_thinking(text, enabled):
    """Split full generated text into (reasoning, content) for JSON mode."""
    if not enabled:
        return "", text
    stripped = text.lstrip()
    if stripped.startswith(_THINK_OPEN):
        text = text[:len(text) - len(stripped)] + stripped[len(_THINK_OPEN):]
    idx = text.find(_THINK_CLOSE)
    if idx == -1:
        return "", text  # unclosed: fail open to content so answers never vanish
    return text[:idx].strip(), text[idx + len(_THINK_CLOSE):].lstrip("\n")


def _normalize_tool_messages(messages):
    """Make client history template-safe in place; returns the same list.

    The pinned template iterates `tool_call.function.arguments|items`, so
    OpenAI string-encoded arguments must be parsed to dicts first. String
    content that is not valid JSON is kept under a single "value" key rather
    than failing the render.
    """
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except Exception:
                    fn["arguments"] = {"value": fn["arguments"]}
    return messages


def _parse_hermes_tool_calls(text, req_id):
    """Split generated text into (prefix_text, openai_tool_calls).

    Returns ([], no calls) when the model answered in plain text. The prefix
    (optional Hermes reasoning before the first call) has <think> blocks
    stripped so clients receive clean content.
    """
    matches = list(_TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return text, []
    calls = []
    for i, m in enumerate(matches):
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append({
            "id": f"call_{req_id}_{i}",
            "type": "function",
            "function": {
                "name": m.group(1).strip(),
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    prefix = _THINK_RE.sub("", text[:matches[0].start()]).strip()
    return prefix, calls


# Unified EOS set — must match is_eos_token() in tools/decode/runtime_258v.h:
# 151643/151645 are the pinned Qwen3.5-MoE checkpoint's true EOS ids;
# 248044/248046 are Qwen3.8-era ids kept for old prompts. A set with only one
# pair runs past the other's EOS (2026-09-25 LENGTH-DIFF).
EOS_TOKEN_IDS = {248044, 248046, 151643, 151645}
FIM_STOP_TOKENS = ["<|fim_middle|>", "<|fim_suffix|>", "<|fim_prefix|>", "<|fim_pad|>", "<|file_sep|>"]
FIM_STOP_TOKEN_IDS = {248060, 248061, 248062, 248063, 248065}


def _log_request_perf(req_id, prompt_tokens, gen_tokens, prefill_s, decode_s, finish_reason, extra=""):
    """One-line live performance log per finished request (stderr).

    pp = prompt processing (prefill) tok/s, tg = token generation (decode)
    tok/s. Same convention as the JSON `timings` (tg excludes the prefill
    token). Guards against zero-duration divisions on tiny prompts.
    """
    pp = prompt_tokens / max(1e-6, prefill_s)
    tg = (gen_tokens - 1) / max(1e-6, decode_s)
    sys.stderr.write(
        f"[Server] {req_id} done: prompt={prompt_tokens}tok pp={pp:.1f} tok/s | "
        f"gen={gen_tokens}tok tg={tg:.1f} tok/s | "
        f"total={prefill_s + decode_s:.2f}s finish={finish_reason}{extra}\n")


# I4.4: in-memory KV prefix cache for multi-turn agent sessions.
PREFIX_MIN_REUSE = 512  # only reuse cached prefixes of at least this length


def _prefix_cache_enabled():
    e = os.environ.get("AINFER_PREFIX_CACHE", "1")
    return not (e.strip() == "0" or e.strip().lower() in ("off", "no", "false"))


def _common_prefix_len(a, b):
    """Length of the longest common token prefix of two id lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class IncrementalDecoder:
    """Incrementally decodes token IDs without partial UTF-8 replacement artifacts."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids = []
        self.decoded_text = ""

    def step(self, token_id):
        self.token_ids.append(token_id)
        full_text = tokenizer_tool.decode(self.tokenizer, self.token_ids, skip_special_tokens=False)
        if full_text.endswith("\ufffd"):
            return ""
        if len(full_text) >= len(self.decoded_text):
            new_text = full_text[len(self.decoded_text):]
            self.decoded_text = full_text
            return new_text
        return ""

    def flush(self):
        full_text = tokenizer_tool.decode(self.tokenizer, self.token_ids, skip_special_tokens=False)
        if len(full_text) > len(self.decoded_text):
            new_text = full_text[len(self.decoded_text):]
            self.decoded_text = full_text
            return new_text
        return ""


class StreamStopBuffer:
    """Buffers minimal trailing text to match stop sequences across token boundaries."""

    def __init__(self, stop_sequences=None):
        self.stop_sequences = [s for s in (stop_sequences or []) if s]
        self.buffer = ""
        self.stopped = False
        self.matched_stop = None

    def append(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self.buffer += text

        if not self.stop_sequences:
            to_emit = self.buffer
            self.buffer = ""
            return to_emit

        # 1. Check if any stop sequence is completely present in buffer
        earliest_idx = -1
        matched_seq = None
        for s in self.stop_sequences:
            idx = self.buffer.find(s)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
                matched_seq = s

        if earliest_idx != -1:
            self.stopped = True
            self.matched_stop = matched_seq
            to_emit = self.buffer[:earliest_idx]
            self.buffer = ""  # discard stop sequence and any trailing text
            return to_emit

        # 2. Check if the suffix of self.buffer is a prefix of any stop sequence
        longest_prefix_len = 0
        for s in self.stop_sequences:
            max_k = min(len(self.buffer), len(s) - 1)
            for k in range(max_k, 0, -1):
                if k > longest_prefix_len and self.buffer.endswith(s[:k]):
                    longest_prefix_len = k
                    break

        if longest_prefix_len > 0:
            emit_end = len(self.buffer) - longest_prefix_len
            to_emit = self.buffer[:emit_end]
            self.buffer = self.buffer[emit_end:]
            return to_emit
        else:
            to_emit = self.buffer
            self.buffer = ""
            return to_emit

    def flush(self) -> str:
        if self.stopped:
            return ""
        to_emit = self.buffer
        self.buffer = ""
        return to_emit


class AInferCtypesBinding:
    """Wrapper around libainfer_258v.so C API."""

    def __init__(self, so_path):
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library not found: {so_path}")
        self.lib = ctypes.CDLL(so_path)

        self.lib.ainfer_create.restype = ctypes.c_void_p
        self.lib.ainfer_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
        self.lib.ainfer_init.restype = ctypes.c_int
        self.lib.ainfer_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.ainfer_prefill.restype = ctypes.c_int
        # I4.4: incremental append prefill over the snapshotted prefix.
        # Present only in rebuilt libainfer_258v.so; absent -> prefix cache
        # stays off (all requests take the full-prefill path).
        try:
            self.lib.ainfer_prefill_incremental.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            self.lib.ainfer_prefill_incremental.restype = ctypes.c_int
            self.lib.ainfer_prefix_cached_len.argtypes = [ctypes.c_void_p]
            self.lib.ainfer_prefix_cached_len.restype = ctypes.c_int
            self.lib.ainfer_prefill_anchor.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            self.lib.ainfer_prefill_anchor.restype = ctypes.c_int
            self.lib.ainfer_prefix_anchor_len.argtypes = [ctypes.c_void_p]
            self.lib.ainfer_prefix_anchor_len.restype = ctypes.c_int
            self.has_prefix_api = True
        except AttributeError:
            self.has_prefix_api = False
        self.lib.ainfer_decode_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.ainfer_decode_step.restype = ctypes.c_int
        self.lib.ainfer_reset_state.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_reset_state.restype = ctypes.c_int
        self.lib.ainfer_get_position.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_get_position.restype = ctypes.c_int
        self.lib.ainfer_is_initialized.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_is_initialized.restype = ctypes.c_int
        self.lib.ainfer_check_device_health.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_check_device_health.restype = ctypes.c_int
        self.lib.ainfer_get_device_name.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_get_device_name.restype = ctypes.c_char_p
        self.lib.ainfer_get_total_memory_bytes.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_get_total_memory_bytes.restype = ctypes.c_uint64
        self.lib.ainfer_init_speculative.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_init_speculative.restype = ctypes.c_int
        self.lib.ainfer_speculative_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ainfer_speculative_step.restype = ctypes.c_int
        self.lib.ainfer_has_speculative.argtypes = [ctypes.c_void_p]
        self.lib.ainfer_has_speculative.restype = ctypes.c_int
        self.lib.ainfer_destroy.argtypes = [ctypes.c_void_p]

        self.handle = self.lib.ainfer_create()
        if not self.handle:
            raise RuntimeError("Failed to instantiate AInferRuntime258V")

    def init(self, model_file, spv_file, max_ctx=4096):
        rc = self.lib.ainfer_init(self.handle, model_file.encode(), spv_file.encode(), max_ctx)
        return rc == 0

    def init_speculative(self):
        return self.lib.ainfer_init_speculative(self.handle) == 0

    def has_speculative(self):
        return bool(self.lib.ainfer_has_speculative(self.handle))

    def prefill(self, prompt_ids):
        c_prompt = (ctypes.c_int * len(prompt_ids))(*prompt_ids)
        out_first = ctypes.c_int(0)
        rc = self.lib.ainfer_prefill(self.handle, c_prompt, len(prompt_ids), ctypes.byref(out_first))
        if rc != 0:
            raise RuntimeError(f"Prefill failed with code {rc}")
        return out_first.value

    def prefill_incremental(self, suffix_ids, start_pos):
        """I4.4: prefill suffix_ids at absolute positions [start_pos, ...).

        Returns the first token, or None when no valid snapshot exists for
        start_pos (or the native API is absent) — caller falls back to full
        prefill. Raises RuntimeError on hard device failure."""
        if not getattr(self, "has_prefix_api", False):
            return None
        c_suffix = (ctypes.c_int * len(suffix_ids))(*suffix_ids)
        out_first = ctypes.c_int(0)
        rc = self.lib.ainfer_prefill_incremental(self.handle, c_suffix, len(suffix_ids), start_pos, ctypes.byref(out_first))
        if rc != 0:
            return None
        return out_first.value

    def prefill_from_anchor(self, full_ids, anchor_pos):
        """I4.4b: rebase full_ids onto the terminal-chunk-boundary anchor.

        Returns the first token, or None when the anchor is stale — caller
        falls back to full prefill."""
        if not getattr(self, "has_prefix_api", False):
            return None
        c_ids = (ctypes.c_int * len(full_ids))(*full_ids)
        out_first = ctypes.c_int(0)
        rc = self.lib.ainfer_prefill_anchor(self.handle, c_ids, len(full_ids), anchor_pos, ctypes.byref(out_first))
        if rc != 0:
            return None
        return out_first.value

    def prefix_anchor_len(self):
        if not getattr(self, "has_prefix_api", False):
            return -1
        return int(self.lib.ainfer_prefix_anchor_len(self.handle))

    def decode_step(self):
        out_next = ctypes.c_int(0)
        rc = self.lib.ainfer_decode_step(self.handle, ctypes.byref(out_next))
        if rc != 0:
            return None
        return out_next.value

    def speculative_step(self):
        t1 = ctypes.c_int(0)
        t2 = ctypes.c_int(0)
        nem = ctypes.c_int(0)
        acc = ctypes.c_int(0)
        rc = self.lib.ainfer_speculative_step(
            self.handle,
            ctypes.byref(t1),
            ctypes.byref(t2),
            ctypes.byref(nem),
            ctypes.byref(acc),
        )
        if rc != 0:
            return None
        return (t1.value, t2.value, nem.value, bool(acc.value))

    def reset_state(self):
        return self.lib.ainfer_reset_state(self.handle) == 0

    def is_initialized(self):
        return bool(self.lib.ainfer_is_initialized(self.handle))

    def check_device_health(self):
        return bool(self.lib.ainfer_check_device_health(self.handle))

    def get_device_name(self):
        return self.lib.ainfer_get_device_name(self.handle).decode()

    def get_total_memory_gib(self):
        return self.lib.ainfer_get_total_memory_bytes(self.handle) / (1024.0 ** 3)

    def close(self):
        if self.handle:
            self.lib.ainfer_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


class ServerState:
    """Manages thread synchronization, bounded queueing, and metrics."""

    def __init__(self, max_queue_capacity=16, default_timeout=60.0):
        self.binding = None
        self.tokenizer = None
        self.worker_lock = threading.Lock()
        self.queue_semaphore = threading.Semaphore(max_queue_capacity)
        self.max_queue_capacity = max_queue_capacity
        self.default_timeout = default_timeout
        self.queue_depth = 0
        self.depth_lock = threading.Lock()
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()
        # I4.4: token ids whose post-prefill KV+SSM state is snapshotted on
        # device (None = no reusable prefix). Only pure appends reuse it.
        self.cached_prompt_ids = None
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.last_prefix_note = ""
        # I4.4b: terminal-chunk-boundary anchor of the cached prompt (None =
        # unknown/stale .so). Chat turns share everything except the trailing
        # generation prompt, so they rebase here instead of exact-extending.
        self.cached_anchor = None


STATE = ServerState()


class AInferHTTPHandler(BaseHTTPRequestHandler):
    server_version = "AInfer/258V"

    def _send_json(self, status_code, data):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Override standard noisy logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

    def is_client_disconnected(self):
        """Check if the client has closed or reset the TCP connection."""
        try:
            sock = getattr(self, "connection", None)
            if sock is None:
                return True
            r, _, _ = select.select([sock], [], [], 0)
            if r:
                data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if len(data) == 0:
                    return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            return True
        return False

    def do_GET(self):
        if self.path == "/healthz":
            self.handle_healthz()
        elif self.path == "/readyz":
            self.handle_readyz()
        elif self.path == "/v1/models":
            self.handle_models()
        else:
            self._send_json(404, {"error": {"message": f"Endpoint not found: {self.path}", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/v1/completions"):
            self.handle_completions(is_chat=(self.path == "/v1/chat/completions"))
        else:
            self._send_json(404, {"error": {"message": f"Endpoint not found: {self.path}", "type": "invalid_request_error"}})

    def handle_healthz(self):
        healthy = STATE.binding.check_device_health() if STATE.binding else False
        with STATE.depth_lock:
            q_depth = STATE.queue_depth

        data = {
            "status": "ok" if healthy else "unhealthy",
            "device": STATE.binding.get_device_name() if STATE.binding else "unknown",
            "device_healthy": healthy,
            "queue_depth": q_depth,
            "max_queue_capacity": STATE.max_queue_capacity,
            "resident_memory_gib": round(STATE.binding.get_total_memory_gib(), 2) if STATE.binding else 0.0,
            "total_requests_served": STATE.total_requests,
            "total_tokens_generated": STATE.total_tokens_generated,
            "uptime_seconds": round(time.time() - STATE.start_time, 1),
            # T8.4: capability marker — remote agents curl this to confirm the
            # live daemon parses Hermes <tool_call> XML into OpenAI tool_calls.
            "tool_calling": "hermes-native",
            "server_build": "258v-t8.4",
            # I4.4: prefix-cache observability.
            "prefix_cache": {
                "enabled": _prefix_cache_enabled(),
                "cached_prompt_tokens": len(STATE.cached_prompt_ids) if STATE.cached_prompt_ids else 0,
                "cached_anchor": STATE.cached_anchor,
                "hits": STATE.prefix_hits,
                "misses": STATE.prefix_misses,
            },
        }
        self._send_json(200 if healthy else 500, data)

    def handle_readyz(self):
        if STATE.binding and STATE.binding.is_initialized() and STATE.binding.check_device_health():
            self._send_json(200, {"status": "ready"})
        else:
            self._send_json(503, {"status": "not_ready"})

    def handle_models(self):
        data = {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(STATE.start_time),
                    "owned_by": "ainfer",
                    "permission": [],
                    "root": MODEL_ID,
                    "parent": None,
                },
                {
                    "id": MODEL_ALIAS,
                    "object": "model",
                    "created": int(STATE.start_time),
                    "owned_by": "ainfer",
                    "permission": [],
                    "root": MODEL_ID,
                    "parent": None,
                }
            ],
        }
        self._send_json(200, data)

    def handle_completions(self, is_chat=True):
        # T8.2: Bounded request queue check
        acquired_queue = STATE.queue_semaphore.acquire(blocking=True, timeout=5.0)
        if not acquired_queue:
            self._send_json(429, {
                "error": {
                    "message": "Server request queue capacity exceeded. Please retry later.",
                    "type": "rate_limit_exceeded",
                    "code": 429
                }
            })
            return

        with STATE.depth_lock:
            STATE.queue_depth += 1

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json(400, {"error": {"message": "Empty request body", "type": "invalid_request_error"}})
                return

            raw_body = self.rfile.read(content_length)
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Malformed JSON: {str(e)}", "type": "invalid_request_error"}})
                return

            raw_stop = body.get("stop")
            if raw_stop is None:
                raw_stop = body.get("stop_sequences")
            stop_sequences = []
            if isinstance(raw_stop, str):
                if raw_stop:
                    stop_sequences.append(raw_stop)
            elif isinstance(raw_stop, list):
                for s in raw_stop:
                    if isinstance(s, str) and s:
                        stop_sequences.append(s)

            stream = bool(body.get("stream", False))
            timeout_s = float(body.get("timeout", STATE.default_timeout))

            # Render Prompt
            try:
                enable_thinking = False
                if is_chat:
                    is_fim = False
                    messages = body.get("messages", [])
                    if not messages or not isinstance(messages, list):
                        self._send_json(400, {"error": {"message": "Field 'messages' must be a non-empty array", "type": "invalid_request_error"}})
                        return
                    # T8.4: forward OpenAI-style tools to the Hermes template
                    # (tool_choice "none" disables; forced function names are
                    # passed through as available tools — no server-side force).
                    raw_tools = body.get("tools")
                    tool_choice = body.get("tool_choice", "auto")
                    tools_for_template = None
                    parse_tools = False
                    if isinstance(raw_tools, list) and raw_tools and tool_choice != "none":
                        tools_for_template = raw_tools
                        parse_tools = True
                    messages = _normalize_tool_messages(messages)
                    # I4.3: native reasoning defaults ON for chat; the pinned
                    # template then ends the prompt with an open `<think>`.
                    enable_thinking = _resolve_enable_thinking(body, is_chat=True)
                    try:
                        prompt_text = tokenizer_tool.render_chat(messages, add_generation_prompt=True, enable_thinking=enable_thinking, tools=tools_for_template)
                    except Exception:
                        prompt_text = ""
                        for m in messages:
                            prompt_text += f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n"
                        prompt_text += "<|im_start|>assistant\n"
                        if enable_thinking:
                            prompt_text += "<think>\n"
                    stop_token_ids = set(EOS_TOKEN_IDS)
                else:
                    prompt = body.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = " ".join(prompt)
                    if not isinstance(prompt, str):
                        self._send_json(400, {"error": {"message": "Field 'prompt' must be a string", "type": "invalid_request_error"}})
                        return

                    suffix = body.get("suffix")
                    if isinstance(suffix, list):
                        suffix = " ".join(suffix)

                    is_fim = False
                    if suffix is not None and isinstance(suffix, str):
                        is_fim = True
                        if "<|fim_middle|>" in prompt or "<|fim_prefix|>" in prompt:
                            prompt_text = prompt
                        else:
                            prompt_text = f"<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"
                    elif "<|fim_middle|>" in prompt:
                        is_fim = True
                        prompt_text = prompt
                    else:
                        prompt_text = prompt

                    if not prompt_text:
                        self._send_json(400, {"error": {"message": "Field 'prompt' must be a non-empty string", "type": "invalid_request_error"}})
                        return

                    stop_token_ids = set(EOS_TOKEN_IDS)
                    if is_fim:
                        stop_token_ids.update(FIM_STOP_TOKEN_IDS)
                        for fim_tok in FIM_STOP_TOKENS:
                            if fim_tok not in stop_sequences:
                                stop_sequences.append(fim_tok)

                max_tokens_default = 256 if (not is_chat and is_fim) else 64
                max_tokens = int(body.get("max_tokens", max_tokens_default))
                max_tokens = max(1, min(max_tokens, 2048))

                prompt_ids = tokenizer_tool.encode(STATE.tokenizer, prompt_text)
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Tokenization/template error: {str(e)}", "type": "invalid_request_error"}})
                return

            # T8.2: Serialize generation through single-flight worker lock
            with STATE.worker_lock:
                self._execute_generation(
                    prompt_ids=prompt_ids,
                    max_tokens=max_tokens,
                    stream=stream,
                    timeout_s=timeout_s,
                    is_chat=is_chat,
                    stop_sequences=stop_sequences,
                    stop_token_ids=stop_token_ids,
                    parse_tools=parse_tools if is_chat else False,
                    enable_thinking=enable_thinking,
                )

        finally:
            with STATE.depth_lock:
                STATE.queue_depth -= 1
            STATE.queue_semaphore.release()

    def _execute_generation(self, prompt_ids, max_tokens, stream, timeout_s, is_chat, stop_sequences=None, stop_token_ids=None, parse_tools=False, enable_thinking=False):
        # T8.2 & T8.3: Check device health before running
        if not STATE.binding.check_device_health():
            self._send_json(500, {"error": {"message": "Level Zero device reported failure or loss", "type": "device_error"}})
            return

        req_id = f"chatcmpl-{int(time.time() * 1000)}" if is_chat else f"cmpl-{int(time.time() * 1000)}"
        created_time = int(time.time())

        # I4.4: prefix-cache dispatch. Two reuse shapes:
        #   exact  — new prompt pure-extends cached ids (L == len cached):
        #            restores the end snapshot, forwards only the suffix.
        #   anchor — new prompt shares L >= 512 tokens with cached ids and L
        #            covers the cached terminal-chunk boundary anchor: restores
        #            the anchor snapshot, reforwards new[anchor..] (<= 1 chunk
        #            of recompute + new tokens). This is the multi-turn chat
        #            shape: the cached trailing generation prompt is replaced
        #            by the assistant turn, so exact can never hit for chat.
        # Anything else (diverged history, short prompt, disabled, stale .so)
        # takes the full-prefill path with a clean reset.
        cached = STATE.cached_prompt_ids
        anchor = STATE.cached_anchor
        prefix_len = _common_prefix_len(cached, prompt_ids) if cached else 0
        use_exact = (
            _prefix_cache_enabled()
            and cached is not None
            and prefix_len == len(cached) >= PREFIX_MIN_REUSE
            and prefix_len < len(prompt_ids)
        )
        use_anchor = (
            not use_exact
            and _prefix_cache_enabled()
            and cached is not None
            and anchor is not None and anchor >= 0
            and prefix_len >= PREFIX_MIN_REUSE
            and anchor <= prefix_len < len(prompt_ids)
            and anchor < len(prompt_ids)
        )
        if use_exact:
            STATE.last_prefix_note = f" prefix={prefix_len}+{len(prompt_ids) - prefix_len}"
        elif use_anchor:
            STATE.last_prefix_note = f" anchor={anchor}+{len(prompt_ids) - anchor}"
        else:
            STATE.last_prefix_note = " full"
            # Ensure state is clean before a full prefill
            STATE.binding.reset_state()

        try:
            t_start = time.perf_counter()
            # 1. Prefill step (blocking; ~8 s per 1K prompt tokens, so a big
            # agentic tools+history prompt can prefill for minutes). In stream
            # mode the SSE headers + role delta are sent first and a keepalive
            # thread emits `: keep-alive` comments during prefill so
            # client/proxy idle-read timeouts don't kill the connection
            # before the first token (I4.2).
            prefill_done = threading.Event()
            if stream:
                # I4.2: headers + role delta go out BEFORE the blocking
                # prefill, so the client sees an immediate TTFT
                # acknowledgment; a daemon thread then emits SSE comment
                # keepalives every 2.5 s during prefill to reset client
                # socket read timeouts (OpenCode/fetch/axios ~30 s).
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.close_connection = True
                if is_chat:
                    try:
                        self.wfile.write(
                            f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return

                def _prefill_keepalive():
                    while not prefill_done.wait(2.5):
                        try:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break

                ka_thread = threading.Thread(target=_prefill_keepalive, daemon=True)
                ka_thread.start()
            try:
                if use_exact:
                    suffix = prompt_ids[prefix_len:]
                    first_tok = STATE.binding.prefill_incremental(suffix, prefix_len)
                    if first_tok is None:
                        # Stale snapshot (or stale .so): fall back to full.
                        sys.stderr.write(f"[Server] Prefix miss ({req_id}): snapshot for {prefix_len} unavailable, full prefill\n")
                        STATE.prefix_misses += 1
                        STATE.last_prefix_note = " full-after-miss"
                        STATE.binding.reset_state()
                        first_tok = STATE.binding.prefill(prompt_ids)
                    else:
                        STATE.prefix_hits += 1
                        sys.stderr.write(f"[Server] Prefix hit ({req_id}): reused {prefix_len}tok, prefilling {len(suffix)}tok\n")
                elif use_anchor:
                    first_tok = STATE.binding.prefill_from_anchor(prompt_ids, anchor)
                    if first_tok is None:
                        sys.stderr.write(f"[Server] Prefix miss ({req_id}): anchor {anchor} unavailable, full prefill\n")
                        STATE.prefix_misses += 1
                        STATE.last_prefix_note = " full-after-miss"
                        STATE.binding.reset_state()
                        first_tok = STATE.binding.prefill(prompt_ids)
                    else:
                        STATE.prefix_hits += 1
                        sys.stderr.write(f"[Server] Anchor hit ({req_id}): shared {prefix_len}tok, anchor {anchor}, prefilling {len(prompt_ids) - anchor}tok\n")
                else:
                    if cached is not None and _prefix_cache_enabled():
                        STATE.prefix_misses += 1
                    first_tok = STATE.binding.prefill(prompt_ids)
                # Prefill succeeded: device now holds exactly prompt_ids.
                STATE.cached_prompt_ids = list(prompt_ids)
                STATE.cached_anchor = STATE.binding.prefix_anchor_len()
            finally:
                if stream:
                    prefill_done.set()
                    ka_thread.join(timeout=20.0)
            t_prefill = time.perf_counter()

            if self.is_client_disconnected():
                sys.stderr.write(f"[Server] Client disconnected after prefill ({req_id}). Aborting.\n")
                return

            if stream:
                self._stream_response(req_id, created_time, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat, len(prompt_ids), stop_sequences, stop_token_ids, parse_tools, headers_sent=True, enable_thinking=enable_thinking)
            else:
                self._complete_response(req_id, created_time, prompt_ids, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat, stop_sequences, stop_token_ids, parse_tools, enable_thinking=enable_thinking)

            STATE.total_requests += 1

        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write(f"[Server] Client disconnected during request {req_id}. Aborting and resetting state.\n")
            # I4.4: device state is indeterminate after an abort — drop cache.
            STATE.cached_prompt_ids = None
            STATE.cached_anchor = None
            try:
                STATE.binding.reset_state()
            except Exception:
                pass
        except Exception as e:
            sys.stderr.write(f"[Server] Error during generation: {e}\n")
            # I4.4: prefill/decode failed — cached ids no longer match device.
            STATE.cached_prompt_ids = None
            STATE.cached_anchor = None
            try:
                STATE.binding.reset_state()
            except Exception:
                pass
            if not stream:
                self._send_json(500, {"error": {"message": str(e), "type": "internal_error"}})
            else:
                try:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
        finally:
            # I4.4: NO unconditional reset here — a successful request leaves
            # the KV prefix + post-prefill SSM snapshot on device for the next
            # append. Cache is invalidated only on the exception paths above.
            pass

    def _complete_response(self, req_id, created_time, prompt_ids, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat, stop_sequences=None, stop_token_ids=None, parse_tools=False, enable_thinking=False):
        stop_token_ids = stop_token_ids or EOS_TOKEN_IDS
        stop_sequences = [s for s in (stop_sequences or []) if s]
        decoder = IncrementalDecoder(STATE.tokenizer)

        generated_ids = [first_tok]
        decoder.step(first_tok)

        stopped = False
        finish_reason = "length"
        output_text = None

        if first_tok in stop_token_ids:
            stopped = True
            finish_reason = "stop"
            output_text = ""
        elif stop_sequences:
            earliest_idx = -1
            for s in stop_sequences:
                idx = decoder.decoded_text.find(s)
                if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                    earliest_idx = idx
            if earliest_idx != -1:
                stopped = True
                finish_reason = "stop"
                output_text = decoder.decoded_text[:earliest_idx]

        # Decode loop
        while not stopped and len(generated_ids) < max_tokens:
            if self.is_client_disconnected():
                sys.stderr.write(f"[Server] Client disconnected mid-generation ({req_id}). Aborting decode loop.\n")
                return
            if time.perf_counter() - t_start > timeout_s:
                sys.stderr.write(f"[Server] Execution timeout ({timeout_s}s) exceeded for {req_id}\n")
                break
            if STATE.binding.has_speculative():
                res = STATE.binding.speculative_step()
                if res is None:
                    break
                t1, t2, nem, _ = res
                tokens_to_emit = [t1] if nem == 1 else [t1, t2]
            else:
                nxt = STATE.binding.decode_step()
                if nxt is None:
                    break
                tokens_to_emit = [nxt]

            for tok in tokens_to_emit:
                generated_ids.append(tok)
                decoder.step(tok)

                if tok in stop_token_ids:
                    stopped = True
                    finish_reason = "stop"
                    break

                if stop_sequences:
                    earliest_idx = -1
                    for s in stop_sequences:
                        idx = decoder.decoded_text.find(s)
                        if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                            earliest_idx = idx
                    if earliest_idx != -1:
                        stopped = True
                        finish_reason = "stop"
                        output_text = decoder.decoded_text[:earliest_idx]
                        break

                if len(generated_ids) >= max_tokens:
                    stopped = True
                    finish_reason = "length"
                    break

        t_end = time.perf_counter()
        STATE.total_tokens_generated += len(generated_ids)

        if output_text is None:
            decoder.flush()
            # I4.3: keep special tokens so <think> tags survive for the
            # split; structural markers are stripped after the split instead.
            output_text = tokenizer_tool.decode(STATE.tokenizer, generated_ids, skip_special_tokens=False)

        if is_chat:
            # T8.4: translate native Hermes <tool_call> XML into OpenAI tool_calls.
            # T8.4b: parse UNCONDITIONALLY — some agents inline tool schemas as
            # prompt text and send no `tools` array; the model still emits
            # Hermes XML, which must come back as tool_calls for the loop to
            # function. Plain answers contain no such blocks and pass through.
            # I4.3: split native thinking first; reasoning is preserved in
            # `reasoning_content` instead of being stripped.
            reasoning, content_text = _split_thinking(output_text, enable_thinking)
            reasoning = _strip_trailing_markers(reasoning)
            content_text = _strip_trailing_markers(content_text)
            prefix, tool_calls = _parse_hermes_tool_calls(content_text, req_id)
            if tool_calls:
                message = {"role": "assistant", "content": prefix or None, "tool_calls": tool_calls}
                finish_reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": content_text}
            if reasoning:
                message["reasoning_content"] = reasoning
            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
                "timings": {
                    "prefill_ms": round((t_prefill - t_start) * 1000.0, 2),
                    "total_latency_ms": round((t_end - t_start) * 1000.0, 2),
                    "decode_tokens_per_s": round((len(generated_ids) - 1) / max(1e-5, (t_end - t_prefill)), 2),
                },
            }
            _log_request_perf(req_id, len(prompt_ids), len(generated_ids),
                              t_prefill - t_start, t_end - t_prefill, finish_reason,
                              extra=STATE.last_prefix_note)
        else:
            resp = {
                "id": req_id,
                "object": "text_completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "text": output_text,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
            }
            _log_request_perf(req_id, len(prompt_ids), len(generated_ids),
                              t_prefill - t_start, t_end - t_prefill, finish_reason,
                              extra=STATE.last_prefix_note)

        self._send_json(200, resp)

    def _stream_response(self, req_id, created_time, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat, prompt_len=0, stop_sequences=None, stop_token_ids=None, parse_tools=False, headers_sent=False, enable_thinking=False):
        stop_token_ids = stop_token_ids or EOS_TOKEN_IDS
        stop_sequences = [s for s in (stop_sequences or []) if s]
        decoder = IncrementalDecoder(STATE.tokenizer)
        stop_buffer = StreamStopBuffer(stop_sequences)

        # Establish Server-Sent Events stream (headers may already be sent when
        # the caller emitted prefill keepalives — see _execute_generation).
        if not headers_sent:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True

        def send_sse_chunk(chunk_dict):
            payload = f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        # I4.2: when headers were sent pre-prefill, the role delta went out
        # with them — don't emit it twice.
        if is_chat and not headers_sent:
            send_sse_chunk({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            })

        STATE.total_tokens_generated += 1
        gen_count = 1
        suppressed = False
        stopped = False
        finish_reason = "length"

        think_splitter = ThinkSplitter(enabled=is_chat and enable_thinking)

        def emit_text_chunk(text):
            # I4.3: route through the think splitter first. Reasoning segments
            # always stream (they precede any tool XML); the raw-XML
            # suppression below gates content segments only.
            if not text:
                return
            for _kind, _seg in think_splitter.feed(text):
                emit_chunk_seg(_seg, _kind)

        def emit_chunk_seg(seg, kind):
            nonlocal suppressed, stopped
            if not seg or stopped:
                return
            if kind == "content":
                if not suppressed and _TOOL_START_RE.search(decoder.decoded_text):
                    suppressed = True
                if suppressed:
                    return
            try:
                if is_chat:
                    delta = {"reasoning_content": seg} if kind == "reasoning" else {"content": seg}
                    send_sse_chunk({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    })
                else:
                    send_sse_chunk({
                        "id": req_id,
                        "object": "text_completion.chunk",
                        "created": created_time,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "text": seg, "finish_reason": None}],
                    })
            except (BrokenPipeError, ConnectionResetError, OSError):
                sys.stderr.write(f"[Server] Client disconnected during SSE emission ({req_id}). Aborting stream.\n")
                stopped = True

        # Process first token
        if first_tok in stop_token_ids:
            stopped = True
            finish_reason = "stop"
        else:
            chunk_text = decoder.step(first_tok)
            safe_text = stop_buffer.append(chunk_text)
            emit_text_chunk(safe_text)
            if stop_buffer.stopped:
                stopped = True
                finish_reason = "stop"

        # Decode streaming loop
        cur_tok = first_tok
        while not stopped and gen_count < max_tokens:
            if self.is_client_disconnected():
                sys.stderr.write(f"[Server] Client disconnected mid-stream ({req_id}). Aborting stream loop.\n")
                stopped = True
                break

            if time.perf_counter() - t_start > timeout_s:
                sys.stderr.write(f"[Server] Stream timeout exceeded for {req_id}\n")
                break

            if STATE.binding.has_speculative():
                res = STATE.binding.speculative_step()
                if res is None:
                    break
                t1, t2, nem, _ = res
                tokens_to_emit = [t1] if nem == 1 else [t1, t2]
            else:
                nxt = STATE.binding.decode_step()
                if nxt is None:
                    break
                tokens_to_emit = [nxt]

            for nxt in tokens_to_emit:
                cur_tok = nxt
                gen_count += 1
                STATE.total_tokens_generated += 1

                if cur_tok in stop_token_ids:
                    stopped = True
                    finish_reason = "stop"
                    break

                chunk_text = decoder.step(cur_tok)
                safe_text = stop_buffer.append(chunk_text)
                emit_text_chunk(safe_text)

                if stop_buffer.stopped:
                    stopped = True
                    finish_reason = "stop"
                    break

                if gen_count >= max_tokens:
                    stopped = True
                    finish_reason = "length"
                    break

        if stopped and self.is_client_disconnected():
            return

        # Flush decoder and stop buffer if not stopped by a stop sequence
        if not stop_buffer.stopped:
            flushed_text = decoder.flush()
            if flushed_text:
                safe_text = stop_buffer.append(flushed_text)
                emit_text_chunk(safe_text)
            remaining_text = stop_buffer.flush()
            emit_text_chunk(remaining_text)
            # I4.3: release the splitter holdback (split-tag guard chars).
            for _kind, _seg in think_splitter.flush():
                emit_chunk_seg(_seg, _kind)

        if stopped and self.is_client_disconnected():
            return

        # Emit terminal SSE chunk + [DONE]
        if is_chat:
            # T8.4: if the streamed text carries Hermes tool calls, emit them
            # as an OpenAI tool_calls delta before the terminal chunk. (Raw
            # XML already streamed as content chunks above; clients that
            # accumulate deltas still converge on the same calls.)
            # T8.4b: unconditional (see _complete_response) — parse even when
            # the request carried no `tools` array.
            _, stream_tool_calls = _parse_hermes_tool_calls(decoder.decoded_text, req_id)
            if stream_tool_calls:
                send_sse_chunk({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "id": tc["id"], "type": "function",
                         "function": {"name": tc["function"]["name"],
                                      "arguments": tc["function"]["arguments"]}}
                        for i, tc in enumerate(stream_tool_calls)]}, "finish_reason": None}],
                })
                finish_reason = "tool_calls"
            send_sse_chunk({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            })
        else:
            send_sse_chunk({
                "id": req_id,
                "object": "text_completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
            })

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        # Live pp/tg performance log (stderr) for the finished stream.
        t_end = time.perf_counter()
        _log_request_perf(req_id, prompt_len, gen_count,
                          t_prefill - t_start, t_end - t_prefill, finish_reason,
                          extra=STATE.last_prefix_note)


def main():
    parser = argparse.ArgumentParser(description="AInfer Resident HTTP Server for Intel Arc 140V (258V)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .binfer model container")
    parser.add_argument("--spv", default=SPV_PATH, help="Path to compiled SPIR-V kernels")
    parser.add_argument("--max-ctx", type=int, default=4096, help="Maximum context tokens (default: 4096)")
    parser.add_argument("--queue-size", type=int, default=16, help="Max waiting queue depth (default: 16)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Execution timeout in seconds (default: 60)")
    parser.add_argument("--speculative", action=argparse.BooleanOptionalAction, default=True, help="Enable dual-token MTP speculative decoding verification (default: True)")
    parser.add_argument("--fast-load", action=argparse.BooleanOptionalAction, default=True, help="Use verified cache stamp to skip redundant 19 GiB CRC re-scan (default: True)")
    parser.add_argument("--verify", action="store_true", help="Force full 19 GiB payload CRC verification on startup")
    args = parser.parse_args()

    if args.verify:
        os.environ["AINFER_VERIFY_CRC"] = "1"
        os.environ["AINFER_FAST_LOAD"] = "0"
    elif args.fast_load:
        os.environ["AINFER_FAST_LOAD"] = "1"

    print("=================================================================")
    print("--- AInfer Resident In-Process HTTP Server Daemon (Xe2 258V) ---")
    print("=================================================================")
    print(f"  Model Container:   {args.model}")
    print(f"  SPIR-V Module:     {args.spv}")
    print(f"  Max Context:       {args.max_ctx} tokens")
    print(f"  Speculative Mode:  {'Enabled' if args.speculative else 'Disabled'}")
    print(f"  Queue Capacity:    {args.queue_size} concurrent requests")
    print(f"  Default Timeout:   {args.timeout} s")
    print(f"  Binding Address:   http://{args.host}:{args.port}")
    print("-----------------------------------------------------------------")

    # 1. Load Tokenizer
    print("[1/3] Loading tokenizer...")
    STATE.tokenizer = tokenizer_tool.load()

    # 2. Instantiate and Initialize Level Zero Runtime
    print("[2/3] Initializing resident Level Zero runtime and uploading weights...")
    STATE.max_queue_capacity = args.queue_size
    STATE.default_timeout = args.timeout
    STATE.queue_semaphore = threading.Semaphore(args.queue_size)

    STATE.binding = AInferCtypesBinding(SO_PATH)
    success = STATE.binding.init(args.model, args.spv, max_ctx=args.max_ctx)
    if not success:
        print("FATAL: Failed to initialize AInfer Level Zero runtime")
        sys.exit(1)

    if args.speculative:
        print("  Initializing MTP speculative verification engine (T10.1)...")
        if STATE.binding.init_speculative():
            print("  Speculative verification: ENABLED (Dual-token verification active)")
        else:
            print("  WARNING: Failed to initialize speculative verification; falling back to autoregressive decode")

    print(f"  Device:            {STATE.binding.get_device_name()}")
    print(f"  Resident Memory:   {STATE.binding.get_total_memory_gib():.2f} GiB")
    print(f"  Device Healthy:    {STATE.binding.check_device_health()}")

    # 3. Start HTTP Server
    print(f"[3/3] Starting HTTP server on {args.host}:{args.port}...")
    server = ThreadingHTTPServer((args.host, args.port), AInferHTTPHandler)
    print("\nServer is READY to accept requests without reloading model weights or rebuilding command lists.")
    print(f"Endpoints:\n  http://{args.host}:{args.port}/healthz\n  http://{args.host}:{args.port}/readyz\n  http://{args.host}:{args.port}/v1/models\n  http://{args.host}:{args.port}/v1/chat/completions\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()
        if STATE.binding:
            STATE.binding.close()
        print("AInfer HTTP daemon stopped cleanly.")


if __name__ == "__main__":
    main()
