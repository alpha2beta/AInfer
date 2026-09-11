"""T7.3 OpenAI-compatible HTTP daemon over the recorded loop (decode_l0).

Endpoints (stdlib only, no dependencies):
  GET  /healthz        -> {"status": "ok"}
  GET  /v1/models      -> {"data": [{"id": <model>, ...}]}
  POST /v1/completions -> {prompt, max_tokens, temperature, top_k, top_p,
                           seed, rep_penalty, stream}
    stream=false: full completion JSON after generation.
    stream=true:  SSE `data:` chunks per generated token + `data: [DONE]`.
    Real streaming: decode_l0 runs with unbuffered stdout and the daemon
    parses its per-step `token %d` lines as they arrive.

Single-flight worker lock (batch-1 device); concurrent requests serialize.
Each request spawns one decode_l0 process (model load ~44 s from USB is the
honest TTFT cost; a persistent worker is scoped follow-on work).
Usage: ainfer_http.py [--port 8010]
"""
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
import tok as tokenizer

BIN = os.path.join(REPO, "build-b60/tools/decode/decode_l0")
MODEL = os.path.join(REPO, "models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer")
SPVDIR = os.path.join(REPO, "build-b60/tools/cmdlist")
ONEAPI = "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; "
MODEL_ID = "qwen3.8-27b-text-int4g128"
STEP_RE = re.compile(r"^step \d+ pos \d+ -> token (\d+)")

TOK = tokenizer.load()
WORKER = threading.Lock()


def generate(ids, max_new, samp, on_token=None):
    """Run one completion. on_token(id) called per generated token as the
    binary emits step lines (None disables streaming reads). Returns
    (ids_generated, wall_s). Raises RuntimeError on nonzero exit."""
    import time
    args = [f"--ids={','.join(map(str, ids))}", f"--max-new={max_new}"]
    if samp.get("temperature", 0) > 0:
        args += [f"--temp={samp['temperature']}",
                 f"--top-k={samp.get('top_k', 0)}",
                 f"--top-p={samp.get('top_p', 1.0)}",
                 f"--seed={samp.get('seed', 0)}",
                 f"--rep-penalty={samp.get('rep_penalty', 1.0)}"]
    cmd = (f"{ONEAPI}stdbuf -o0 -e0 {BIN} {MODEL} {len(ids)} 1 "
           f"{SPVDIR} /dev/stdout {' '.join(args)}")
    t0 = time.time()
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    gen, tail = [], ""
    for line in p.stdout:
        m = STEP_RE.match(line.strip())
        if m:
            tid = int(m.group(1))
            gen.append(tid)
            if on_token:
                on_token(tid)
        if line.startswith('{"device"'):
            tail = line
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"decode_l0 rc={rc}")
    return gen, time.time() - t0


class Handler(BaseHTTPRequestHandler):
    server_version = "AInfer/1.0"

    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": MODEL_ID, "owned_by": "ainfer"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, {"error": "invalid JSON"})
            return
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        if not isinstance(prompt, str) or not prompt:
            self._send(400, {"error": "missing prompt"})
            return
        try:
            max_new = int(body.get("max_tokens", 16))
        except Exception:
            self._send(400, {"error": "bad max_tokens"})
            return
        max_new = max(1, min(max_new, 512))
        samp = {"temperature": float(body.get("temperature", 0) or 0),
                "top_k": int(body.get("top_k", 0) or 0),
                "top_p": float(body.get("top_p", 1.0) or 1.0),
                "seed": int(body.get("seed", 0) or 0),
                "rep_penalty": float(body.get("rep_penalty", 1.0) or 1.0)}
        try:
            ids = tokenizer.encode(TOK, prompt)
        except Exception as e:
            self._send(400, {"error": f"tokenize failed: {e}"})
            return
        if body.get("stream"):
            self._stream(ids, max_new, samp, prompt)
        else:
            self._complete(ids, max_new, samp, prompt)

    def _complete(self, ids, max_new, samp, prompt):
        try:
            with WORKER:
                gen, wall = generate(ids, max_new, samp)
        except RuntimeError as e:
            self._send(500, {"error": str(e)})
            return
        text = tokenizer.decode(TOK, gen)
        self._send(200, {"id": "ainfer-0", "model": MODEL_ID,
                         "choices": [{"text": text, "index": 0,
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": len(ids),
                                   "completion_tokens": len(gen)},
                         "timings": {"wall_s": round(wall, 1)}})

    def _stream(self, ids, max_new, samp, prompt):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            line = f"data: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(line)
            self.wfile.flush()

        got = []
        try:
            with WORKER:
                def on_token(tid):
                    got.append(tid)
                    emit({"id": "ainfer-0", "model": MODEL_ID,
                          "choices": [{"text": tokenizer.decode(TOK, [tid]),
                                       "index": 0, "finish_reason": None}]})
                generate(ids, max_new, samp, on_token=on_token)
        except (RuntimeError, BrokenPipeError) as e:
            try:
                emit({"error": str(e)})
            except BrokenPipeError:
                pass
            return
        try:
            emit({"id": "ainfer-0", "model": MODEL_ID,
                  "choices": [{"text": "", "index": 0,
                               "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass


if __name__ == "__main__":
    port = 8010
    for i, a in enumerate(sys.argv[1:]):
        if a == "--port" and i + 2 <= len(sys.argv[1:]):
            port = int(sys.argv[i + 2])
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ainfer_http on 127.0.0.1:{port}", flush=True)
    srv.serve_forever()
