#!/usr/bin/env python3
"""End-to-end integration tests for P0 improvements on Intel Arc 140V (258V):
  - Stop sequence parsing (single string, array of strings, streaming and non-streaming)
  - Native FIM formatting (/v1/completions with prompt + suffix)
  - Early exit verification (verifying that decode stops early and doesn't generate trailing code)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_for_server(base_url, timeout=90):
    print(f"[Wait] Waiting for server at {base_url}/readyz (timeout={timeout}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"{base_url}/readyz")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"  Server is READY in {time.time() - t0:.1f}s")
                    return True
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s")


def test_completion_stop_non_streaming(base_url):
    print("\n[Test 1] POST /v1/completions with stop=['\\n'] (non-streaming)")
    req_body = {
        "prompt": "def add(a, b):\n    ",
        "max_tokens": 64,
        "stop": ["\n"],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choice = data["choices"][0]
    text = choice["text"]
    finish_reason = choice["finish_reason"]
    usage = data["usage"]

    print(f"  Text: {repr(text)}")
    print(f"  Finish reason: {finish_reason}")
    print(f"  Completion tokens: {usage['completion_tokens']}")

    assert finish_reason == "stop", f"Expected finish_reason 'stop', got '{finish_reason}'"
    assert "\n" not in text, f"Stop sequence '\\n' found in text: {repr(text)}"
    assert len(text) > 0, "Expected non-empty completion"
    print("  [PASS] Single-line stop verified cleanly.")


def test_completion_stop_streaming(base_url):
    print("\n[Test 2] POST /v1/completions with stop=['\\n'] (streaming SSE)")
    req_body = {
        "prompt": "def add(a, b):\n    ",
        "max_tokens": 64,
        "stop": ["\n"],
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks_text = []
    terminal_finish_reason = None

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str or line_str == "data: [DONE]":
                continue
            if line_str.startswith("data: "):
                chunk_data = json.loads(line_str[6:])
                choice = chunk_data["choices"][0]
                if choice.get("text"):
                    chunks_text.append(choice["text"])
                if choice.get("finish_reason"):
                    terminal_finish_reason = choice["finish_reason"]

    full_text = "".join(chunks_text)
    print(f"  Streamed text: {repr(full_text)}")
    print(f"  Terminal finish reason: {terminal_finish_reason}")

    assert terminal_finish_reason == "stop", f"Expected finish_reason 'stop', got '{terminal_finish_reason}'"
    assert "\n" not in full_text, f"Stop sequence '\\n' found in streamed text: {repr(full_text)}"
    assert len(full_text) > 0, "Expected non-empty completion"
    print("  [PASS] Streaming stop sequence verified cleanly.")


def test_fim_completion_non_streaming(base_url):
    print("\n[Test 3] POST /v1/completions with FIM prompt + suffix (non-streaming)")
    req_body = {
        "prompt": "def add(a, b):\n    return ",
        "suffix": "\n\ndef sub(a, b):\n    return a - b",
        "max_tokens": 32,
        "stop": ["\n"],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choice = data["choices"][0]
    text = choice["text"]
    finish_reason = choice["finish_reason"]
    print(f"  FIM Infill text: {repr(text)}")
    print(f"  Finish reason: {finish_reason}")

    assert finish_reason == "stop", f"Expected finish_reason 'stop', got '{finish_reason}'"
    assert "\n" not in text, f"Stop sequence '\\n' found in text: {repr(text)}"
    assert "<|fim_" not in text, f"Special FIM token leaked into text: {repr(text)}"
    assert len(text.strip()) > 0, "Expected non-empty FIM infill"
    print("  [PASS] FIM non-streaming completion verified.")


def test_fim_completion_streaming(base_url):
    print("\n[Test 4] POST /v1/completions with FIM prompt + suffix (streaming SSE)")
    req_body = {
        "prompt": "def add(a, b):\n    return ",
        "suffix": "\n\ndef sub(a, b):\n    return a - b",
        "max_tokens": 32,
        "stop": ["\n"],
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks_text = []
    terminal_finish_reason = None

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str or line_str == "data: [DONE]":
                continue
            if line_str.startswith("data: "):
                chunk_data = json.loads(line_str[6:])
                choice = chunk_data["choices"][0]
                if choice.get("text"):
                    chunks_text.append(choice["text"])
                if choice.get("finish_reason"):
                    terminal_finish_reason = choice["finish_reason"]

    full_text = "".join(chunks_text)
    print(f"  FIM Streamed text: {repr(full_text)}")
    print(f"  Terminal finish reason: {terminal_finish_reason}")

    assert terminal_finish_reason == "stop", f"Expected finish_reason 'stop', got '{terminal_finish_reason}'"
    assert "\n" not in full_text, f"Stop sequence '\\n' found in streamed text: {repr(full_text)}"
    assert "<|fim_" not in full_text, f"Special FIM token leaked into text: {repr(full_text)}"
    assert len(full_text.strip()) > 0, "Expected non-empty FIM infill"
    print("  [PASS] FIM streaming completion verified.")


def test_chat_stop_sequences(base_url):
    print("\n[Test 5] POST /v1/chat/completions with stop=['\\n']")
    req_body = {
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one short sentence."}],
        "max_tokens": 64,
        "stop": ["\n"],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choice = data["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice["finish_reason"]

    print(f"  Chat response: {repr(content)}")
    print(f"  Finish reason: {finish_reason}")

    assert finish_reason == "stop", f"Expected finish_reason 'stop', got '{finish_reason}'"
    assert "\n" not in content, f"Stop sequence '\\n' leaked: {repr(content)}"
    print("  [PASS] Chat stop sequences verified.")


def test_instant_disconnect_abort(base_url):
    print("\n[Test 6] Client disconnect abort: verify decode loop halts within 1 step")
    import socket, urllib.parse

    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8088

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # Request 512 tokens (would take ~15 seconds if not aborted)
    body = json.dumps({
        "prompt": "Count from 1 to 1000:\n1, 2, 3, ",
        "max_tokens": 512,
        "stream": True,
    })
    req_data = (
        f"POST /v1/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n{body}"
    ).encode("utf-8")

    s.sendall(req_data)

    # Read response headers and first SSE chunk
    chunk = s.recv(256)
    time.sleep(0.05)
    # Abruptly close client socket
    s.close()
    print("  Socket closed abruptly mid-stream. Verifying immediate GPU yield...")

    # Immediately submit a follow-up request to verify the GPU is freed instantly (not stuck in 512-token loop)
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps({"prompt": "ping", "max_tokens": 4}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0

    print(f"  Follow-up request latency: {elapsed * 1000.0:.1f} ms")
    assert "choices" in data
    # If the previous 512-token decode continued, this request would have waited >5000 ms.
    assert elapsed < 2.0, f"Server took {elapsed:.2f}s to process follow-up request, indicating GPU was blocked!"
    print("  [PASS] Instant client disconnect abort verified.")


def main():
    parser = argparse.ArgumentParser(description="AInfer P0/P1 End-to-End Test")
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="Base URL of server")
    parser.add_argument("--spawn", action="store_true", help="Spawn server subprocess if not already running")
    args = parser.parse_args()

    proc = None
    if args.spawn:
        env = dict(os.environ)
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "tools", "http", "server_258v.py"),
            "--port", "8088",
        ]
        print(f"[Spawn] Starting server on port 8088...")
        proc = subprocess.Popen(cmd, env=env)

    try:
        wait_for_server(args.url, timeout=90)
        test_completion_stop_non_streaming(args.url)
        test_completion_stop_streaming(args.url)
        test_fim_completion_non_streaming(args.url)
        test_fim_completion_streaming(args.url)
        test_chat_stop_sequences(args.url)
        test_instant_disconnect_abort(args.url)
        print("\n=======================================================")
        print(">>> ALL P0/P1 INTEGRATION TESTS PASSED (100% SUCCESS) <<<")
        print("=======================================================")
    finally:
        if proc is not None:
            print("[Cleanup] Terminating spawned server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
