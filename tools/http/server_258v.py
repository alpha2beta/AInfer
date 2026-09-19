#!/usr/bin/env python3
"""Task T8.1 - T8.3: Persistent Resident In-Process HTTP Server for Intel Arc 140V (258V).

Maintains model weights and Level Zero recorded command lists permanently
resident in unified system memory across requests (Zero model reloading, zero
command list rebuilding).

Endpoints:
  GET  /healthz              -> Runtime and device health status
  GET  /readyz               -> Readiness check
  GET  /v1/models            -> Model listing
  POST /v1/chat/completions  -> OpenAI-compatible chat completion (streaming SSE & full JSON)
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
        self.lib.ainfer_destroy.argtypes = [ctypes.c_void_p]

        self.handle = self.lib.ainfer_create()
        if not self.handle:
            raise RuntimeError("Failed to instantiate AInferRuntime258V")

    def init(self, model_file, spv_file, max_ctx=4096):
        rc = self.lib.ainfer_init(self.handle, model_file.encode(), spv_file.encode(), max_ctx)
        return rc == 0

    def prefill(self, prompt_ids):
        c_prompt = (ctypes.c_int * len(prompt_ids))(*prompt_ids)
        out_first = ctypes.c_int(0)
        rc = self.lib.ainfer_prefill(self.handle, c_prompt, len(prompt_ids), ctypes.byref(out_first))
        if rc != 0:
            raise RuntimeError(f"Prefill failed with code {rc}")
        return out_first.value

    def decode_step(self):
        out_next = ctypes.c_int(0)
        rc = self.lib.ainfer_decode_step(self.handle, ctypes.byref(out_next))
        if rc != 0:
            return None
        return out_next.value

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

            stream = bool(body.get("stream", False))
            max_tokens = int(body.get("max_tokens", 64))
            max_tokens = max(1, min(max_tokens, 2048))
            timeout_s = float(body.get("timeout", STATE.default_timeout))

            # Render Prompt
            try:
                if is_chat:
                    messages = body.get("messages", [])
                    if not messages or not isinstance(messages, list):
                        self._send_json(400, {"error": {"message": "Field 'messages' must be a non-empty array", "type": "invalid_request_error"}})
                    try:
                        prompt_text = tokenizer_tool.render_chat(messages, add_generation_prompt=True, enable_thinking=False)
                    except Exception:
                        prompt_text = ""
                        for m in messages:
                            prompt_text += f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n"
                        prompt_text += "<|im_start|>assistant\n"
                else:
                    prompt = body.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = " ".join(prompt)
                    if not prompt or not isinstance(prompt, str):
                        self._send_json(400, {"error": {"message": "Field 'prompt' must be a non-empty string", "type": "invalid_request_error"}})
                        return
                    prompt_text = prompt

                prompt_ids = tokenizer_tool.encode(STATE.tokenizer, prompt_text)
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Tokenization/template error: {str(e)}", "type": "invalid_request_error"}})
                return

            # T8.2: Serialize generation through single-flight worker lock
            with STATE.worker_lock:
                self._execute_generation(prompt_ids, max_tokens, stream, timeout_s, is_chat)

        finally:
            with STATE.depth_lock:
                STATE.queue_depth -= 1
            STATE.queue_semaphore.release()

    def _execute_generation(self, prompt_ids, max_tokens, stream, timeout_s, is_chat):
        # T8.2 & T8.3: Check device health before running
        if not STATE.binding.check_device_health():
            self._send_json(500, {"error": {"message": "Level Zero device reported failure or loss", "type": "device_error"}})
            return

        req_id = f"chatcmpl-{int(time.time() * 1000)}" if is_chat else f"cmpl-{int(time.time() * 1000)}"
        created_time = int(time.time())

        # Ensure state is clean before start
        STATE.binding.reset_state()

        try:
            t_start = time.perf_counter()
            # 1. Prefill step
            first_tok = STATE.binding.prefill(prompt_ids)
            t_prefill = time.perf_counter()

            if stream:
                self._stream_response(req_id, created_time, first_tok, max_tokens, t_start, timeout_s, is_chat)
            else:
                self._complete_response(req_id, created_time, prompt_ids, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat)

            STATE.total_requests += 1

        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write(f"[Server] Client disconnected during request {req_id}. Aborting and resetting state.\n")
        except Exception as e:
            sys.stderr.write(f"[Server] Error during generation: {e}\n")
            if not stream:
                self._send_json(500, {"error": {"message": str(e), "type": "internal_error"}})
        finally:
            # T8.2: Always reset state on finish or abort so GPU memory/SSM buffers are clean
            STATE.binding.reset_state()

    def _complete_response(self, req_id, created_time, prompt_ids, first_tok, max_tokens, t_start, t_prefill, timeout_s, is_chat):
        generated_ids = [first_tok]

        # Decode loop
        while len(generated_ids) < max_tokens:
            if time.perf_counter() - t_start > timeout_s:
                sys.stderr.write(f"[Server] Execution timeout ({timeout_s}s) exceeded for {req_id}\n")
                break
            nxt = STATE.binding.decode_step()
            if nxt is None:
                break
            generated_ids.append(nxt)
            # Break on EOS tokens (248044, 248046)
            if nxt in (248044, 248046):
                break

        t_end = time.perf_counter()
        STATE.total_tokens_generated += len(generated_ids)
        output_text = tokenizer_tool.decode(STATE.tokenizer, generated_ids, skip_special_tokens=True)

        if is_chat:
            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output_text},
                        "finish_reason": "stop" if generated_ids[-1] in (248044, 248046) else "length",
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
                        "finish_reason": "stop" if generated_ids[-1] in (248044, 248046) else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
            }

        self._send_json(200, resp)

    def _stream_response(self, req_id, created_time, first_tok, max_tokens, t_start, timeout_s, is_chat):
        # Establish Server-Sent Events stream
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send_sse_chunk(chunk_dict):
            payload = f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        # Emit first token
        first_text = tokenizer_tool.decode(STATE.tokenizer, [first_tok], skip_special_tokens=False)
        STATE.total_tokens_generated += 1
        gen_count = 1

        if is_chat:
            send_sse_chunk({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": first_text}, "finish_reason": None}],
            })
        else:
            send_sse_chunk({
                "id": req_id,
                "object": "text_completion.chunk",
                "created": created_time,
                "model": MODEL_ID,
                "choices": [{"index": 0, "text": first_text, "finish_reason": None}],
            })

        # Decode streaming loop
        cur_tok = first_tok
        while gen_count < max_tokens:
            if cur_tok in (248044, 248046):
                break
            if time.perf_counter() - t_start > timeout_s:
                sys.stderr.write(f"[Server] Stream timeout exceeded for {req_id}\n")
                break

            nxt = STATE.binding.decode_step()
            if nxt is None:
                break
            cur_tok = nxt
            gen_count += 1
            STATE.total_tokens_generated += 1

            if cur_tok in (248044, 248046):
                break

            chunk_text = tokenizer_tool.decode(STATE.tokenizer, [cur_tok], skip_special_tokens=False)
            if is_chat:
                send_sse_chunk({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                })
            else:
                send_sse_chunk({
                    "id": req_id,
                    "object": "text_completion.chunk",
                    "created": created_time,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "text": chunk_text, "finish_reason": None}],
                })

        # Emit terminal SSE chunk + [DONE]
        finish_reason = "stop" if cur_tok in (248044, 248046) else "length"
        if is_chat:
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


def main():
    parser = argparse.ArgumentParser(description="AInfer Resident HTTP Server for Intel Arc 140V (258V)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .binfer model container")
    parser.add_argument("--spv", default=SPV_PATH, help="Path to compiled SPIR-V kernels")
    parser.add_argument("--max-ctx", type=int, default=4096, help="Maximum context tokens (default: 4096)")
    parser.add_argument("--queue-size", type=int, default=16, help="Max waiting queue depth (default: 16)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Execution timeout in seconds (default: 60)")
    args = parser.parse_args()

    print("=================================================================")
    print("--- AInfer Resident In-Process HTTP Server Daemon (Xe2 258V) ---")
    print("=================================================================")
    print(f"  Model Container:   {args.model}")
    print(f"  SPIR-V Module:     {args.spv}")
    print(f"  Max Context:       {args.max_ctx} tokens")
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
