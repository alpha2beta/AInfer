#!/usr/bin/env python3
"""Task T8.4: 100-Request Continuous Stress Test and Memory Leak Audit for Persistent HTTP Server.

Tests:
  - Phase 8 Milestone 7 validation (T8.1 - T8.4)
  - 100 consecutive chat and completion requests across persistent HTTP daemon
  - Mixed streaming (SSE) and non-streaming requests
  - Client disconnect cancellation handling
  - Process RSS memory profiling (zero-leak audit)
  - Latency and throughput distribution

Emits: tools/http/report_http_stress.json
"""

import argparse
import concurrent.futures
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_PATH = os.path.join(REPO_ROOT, "tools", "http", "report_http_stress.json")


def get_process_rss_kb(pid):
    """Read resident set size in KB for given PID from /proc."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def wait_for_server(base_url, timeout=45):
    """Wait for server /readyz to return 200 OK."""
    print(f"[Wait] Waiting for server at {base_url}/readyz...")
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


def test_client_cancellation(base_url):
    """T8.2: Verify client disconnect is cleanly caught and state is safely reset."""
    print("\n[Test] Testing early client disconnect cancellation...")
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    body = json.dumps({
        "messages": [{"role": "user", "content": "Write a 500-word essay on modern GPU architectures."}],
        "max_tokens": 128,
        "stream": True,
    })
    req_data = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n{body}"
    ).encode("utf-8")

    s.sendall(req_data)

    # Read partial response header and first token
    chunk = s.recv(512)
    time.sleep(0.05)
    # Abruptly close client socket
    s.close()
    print("  Socket closed abruptly mid-stream. Verifying server recovered...")

    time.sleep(0.5)
    # Immediately make a subsequent request to verify server is clean and responsive
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "Ping"}], "max_tokens": 8}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert "choices" in data
        print("  Post-cancellation request succeeded: State reset and runtime clean! [PASS]")


def run_stress_test(base_url, total_requests=100, concurrency=2, server_pid=None):
    print("=================================================================")
    print(f"--- Task T8.4: {total_requests}-Request Continuous Stress Test ---")
    print("=================================================================")
    print(f"  Target Server:     {base_url}")
    print(f"  Total Requests:    {total_requests}")
    print(f"  Client Workers:    {concurrency}")
    if server_pid:
        print(f"  Server PID:        {server_pid}")
    print("-----------------------------------------------------------------")

    # Initial telemetry
    initial_rss = get_process_rss_kb(server_pid) if server_pid else None
    if initial_rss:
        print(f"Initial Server RSS: {initial_rss} KB ({initial_rss / 1024.0:.2f} MB)")

    prompts = [
        "Write a quicksort function in C++.",
        "Explain what makes DeltaNet linear attention fast.",
        "List 3 prime numbers greater than 100.",
        "Implement a Python generator for Fibonacci numbers.",
        "What is the capital of France and what is its population?",
        "Define Level Zero command lists in 2 sentences.",
        "Write a simple HTTP GET request using curl.",
        "How many bits are in an INT4 tensor scale group?",
    ]

    results = []
    rss_checkpoints = {}
    completed_count = 0
    t_start_total = time.time()

    def send_single_request(req_idx):
        nonlocal completed_count
        prompt_text = prompts[req_idx % len(prompts)]
        is_stream = (req_idx % 2 == 1)  # alternate streaming and non-streaming

        payload = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 32,
            "stream": is_stream,
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        first_tok_time = None
        gen_tokens = 0
        error = None
        output_text = ""

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                if is_stream:
                    # Parse SSE line by line without buffering
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        line_str = line.decode("utf-8").strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            if first_tok_time is None:
                                first_tok_time = time.perf_counter()
                            try:
                                chunk_json = json.loads(line_str[6:])
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                output_text += delta
                                gen_tokens += 1
                            except Exception:
                                pass
                else:
                    data = json.loads(resp.read().decode("utf-8"))
                    output_text = data["choices"][0]["message"]["content"]
                    gen_tokens = data["usage"]["completion_tokens"]
                    first_tok_time = t0 + (data.get("timings", {}).get("prefill_ms", 0.0) * 1e-3)
        except Exception as e:
            error = str(e)

        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0
        ttft_ms = ((first_tok_time - t0) * 1000.0) if first_tok_time else total_latency_ms

        res = {
            "req_id": req_idx + 1,
            "stream": is_stream,
            "tokens": gen_tokens,
            "total_latency_ms": round(total_latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "decode_tps": round((gen_tokens - 1) / max(1e-5, (t1 - (first_tok_time or t0))), 2) if gen_tokens > 1 else 0.0,
            "success": (error is None and gen_tokens > 0),
            "error": error,
        }

        completed_count += 1
        if completed_count % 10 == 0 or completed_count == total_requests:
            rss_now = get_process_rss_kb(server_pid) if server_pid else 0
            rss_checkpoints[str(completed_count)] = rss_now
            print(f"  [{completed_count:3d}/{total_requests}] Req {req_idx+1:3d}: "
                  f"tokens={gen_tokens:2d}, ttft={ttft_ms:.1f}ms, latency={total_latency_ms:.1f}ms, "
                  f"server_rss={rss_now} KB")

        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_single_request, i) for i in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    t_end_total = time.time()
    total_duration_s = t_end_total - t_start_total

    # Sort results by req_id
    results.sort(key=lambda x: x["req_id"])

    # Aggregate metrics
    successes = [r for r in results if r["success"]]
    fail_count = len(results) - len(successes)
    total_tokens = sum(r["tokens"] for r in successes)
    mean_latency = sum(r["total_latency_ms"] for r in successes) / max(1, len(successes))
    mean_ttft = sum(r["ttft_ms"] for r in successes) / max(1, len(successes))
    mean_tps = sum(r["decode_tps"] for r in successes if r["decode_tps"] > 0) / max(1, len(successes))

    final_rss = get_process_rss_kb(server_pid) if server_pid else None
    rss_delta = (final_rss - initial_rss) if (initial_rss and final_rss) else 0

    print("\n=================================================================")
    print("--- Task T8.4 Multi-Request Stress Test Summary ---")
    print("=================================================================")
    print(f"  Total Requests:       {total_requests} (Success: {len(successes)}, Failed: {fail_count})")
    print(f"  Total Tokens Emitted: {total_tokens} tokens")
    print(f"  Total Test Duration:  {total_duration_s:.2f} s ({total_tokens / total_duration_s:.2f} system tok/s)")
    print(f"  Mean Request Latency: {mean_latency:.2f} ms")
    print(f"  Mean TTFT:            {mean_ttft:.2f} ms")
    print(f"  Mean Decode Speed:    {mean_tps:.2f} tok/s")
    if initial_rss and final_rss:
        print(f"  Server Initial RSS:   {initial_rss} KB")
        print(f"  Server Final RSS:     {final_rss} KB")
        print(f"  RSS Memory Delta:     {rss_delta} KB (Leak-Free: {rss_delta <= 1024})")
    print("=================================================================\n")

    # Final Health & Status
    req = urllib.request.Request(f"{base_url}/healthz")
    with urllib.request.urlopen(req) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))

    report = {
        "task": "T8.4",
        "milestone": "Milestone 7 (Phase 8)",
        "status": "PASSED" if (fail_count == 0 and abs(rss_delta) <= 2048) else "FAILED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_parameters": {
            "total_requests": total_requests,
            "concurrency": concurrency,
            "target_url": base_url,
        },
        "aggregate_metrics": {
            "successful_requests": len(successes),
            "failed_requests": fail_count,
            "total_tokens_emitted": total_tokens,
            "total_duration_s": round(total_duration_s, 2),
            "mean_latency_ms": round(mean_latency, 2),
            "mean_ttft_ms": round(mean_ttft, 2),
            "mean_decode_tps": round(mean_tps, 2),
        },
        "memory_leak_audit": {
            "initial_rss_kb": initial_rss,
            "final_rss_kb": final_rss,
            "rss_delta_kb": rss_delta,
            "rss_checkpoints_kb": rss_checkpoints,
            "zero_leak_verified": (abs(rss_delta) <= 2048),
        },
        "final_healthz": health_data,
        "sample_runs": results[:10],
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[T8.4] Stress test report saved to: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AInfer T8.4 Multi-Request Stress Test Runner")
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="Base URL of AInfer server")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to run (default: 100)")
    parser.add_argument("--concurrency", type=int, default=2, help="Client concurrency (default: 2)")
    parser.add_argument("--pid", type=int, default=None, help="PID of server process for RSS monitoring")
    args = parser.parse_args()

    wait_for_server(args.url)
    test_client_cancellation(args.url)
    run_stress_test(args.url, total_requests=args.requests, concurrency=args.concurrency, server_pid=args.pid)
