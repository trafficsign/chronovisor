#!/usr/bin/env python3.14
"""Isolated DwarfStar/oMLX comparison, reusing the existing synthetic tests."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import sawfwair_quant_benchmark as bench
from vq32_benchmark import prompt_schema

WORK = Path("/Users/trafficsign/.omlx/experiments/ds4-qwen38-q4")
ENGINE = Path("/Users/trafficsign/.local/share/dwarfstar/runtime")
MODEL_DIR = Path("/Users/trafficsign/.local/share/dwarfstar/models")
OUT = bench.ROOT / "_handoff/evidence/2026-09-06-ds4-qwen38-benchmark"
MANAGER = "/Users/trafficsign/Applications/Chronovisor.app/Contents/MacOS/Chronovisor"
PROTECTED = [
    Path("/Users/trafficsign/.chronovisor/config.toml"),
    Path("/Users/trafficsign/.omlx/settings.json"),
    Path("/Users/trafficsign/.omlx/model_settings.json"),
]
MODEL = "Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf"
PLE = "Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
SIZES = {MODEL: 74879771648, PLE: 32000157440}


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def stop_orphan(arm):
    """The app CLI can report stopped when its detached backend still listens."""
    if not port_open(18125):
        return
    listeners = subprocess.check_output(
        ["lsof", "-nP", "-t", "-iTCP:18125", "-sTCP:LISTEN"], text=True
    ).split()
    if len(set(listeners)) != 1:
        raise RuntimeError("production listener is not a unique verified orphan")
    pid = int(listeners[0])
    identity = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "ppid=,comm="], text=True
    ).split()
    if identity != ["1", "omlx-server"]:
        raise RuntimeError(f"refusing to stop unexpected listener PID {pid}")
    save(
        f"{arm}-orphan-stop.json",
        {"pid": pid, "identity": identity, "signal": "SIGTERM"},
    )
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not port_open(18125):
            return
        time.sleep(1)
    raise RuntimeError("verified orphan did not stop after SIGTERM")


@contextlib.contextmanager
def isolated(arm, download_pid):
    before = hashes()
    state = json.loads(bench.command(MANAGER, "status"))
    enabled = [s["plist"] for s in state["services"] if s["status"] == "enabled"]
    save(f"{arm}-production-before.json", {"hashes": before, "services": state})
    paused = False
    attempted = []
    try:
        if download_pid:
            command = bench.command("ps", "-o", "command=", "-p", str(download_pid))
            if "huggingface-hub" not in command or "hf download" not in command:
                raise RuntimeError("download PID is not the expected hf process")
            os.kill(download_pid, signal.SIGSTOP)
            paused = True
        for plist in enabled:
            attempted.append(plist)
            bench.command(MANAGER, "unregister-one", plist)
        subprocess.run([bench.CLI, "stop", "--timeout", "60"], check=True, timeout=75)
        time.sleep(5)
        stop_orphan(arm)
        if port_open(18125) or port_open(18136):
            raise RuntimeError("production or experimental port still open")
        yield
    finally:
        restored = []
        try:
            result = subprocess.run(
                [bench.CLI, "start", "--timeout", "300"],
                capture_output=True,
                text=True,
                timeout=320,
            )
            restored.append({"service": "omlx", "returncode": result.returncode})
        except Exception as exc:
            restored.append({"service": "omlx", "error": str(exc)})
        for plist in attempted:
            try:
                result = subprocess.run(
                    [MANAGER, "register-one", plist],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                restored.append({"service": plist, "returncode": result.returncode})
            except Exception as exc:
                restored.append({"service": plist, "error": str(exc)})
        if paused:
            try:
                os.kill(download_pid, signal.SIGCONT)
            except ProcessLookupError:
                restored.append(
                    {
                        "service": "download",
                        "returncode": 0,
                        "note": "download process already exited",
                    }
                )
        health = None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:18125/health", timeout=2
                ) as response:
                    health = json.load(response)
                if health.get("status") == "healthy":
                    break
            except Exception:
                pass
            time.sleep(1)
        after = hashes()
        save(
            f"{arm}-production-restored.json",
            {
                "services": restored,
                "hashes": after,
                "unchanged": before == after,
                "health": health,
            },
        )
        if (
            before != after
            or any(r.get("returncode", 1) != 0 for r in restored)
            or not health
            or health.get("status") != "healthy"
        ):
            raise RuntimeError("restoration requires attention; see evidence")


def server_command(arm):
    if arm == "baseline":
        bench.WORK = WORK
        bench.OUT = OUT
        base = bench.prepare("baseline")
        return [
            bench.CLI,
            "serve",
            "--base-path",
            str(base),
            "--model-dir",
            str(base / "models"),
            "--host",
            "127.0.0.1",
            "--port",
            "18136",
            "--no-hf-cache",
            "--max-concurrent-requests",
            "1",
            "--memory-guard-gb",
            "104",
            "--paged-ssd-cache-dir",
            str(base / "cache"),
            "--paged-ssd-cache-max-size",
            "5GB",
            "--hot-cache-max-size",
            "2GB",
        ]
    for name, size in SIZES.items():
        path = MODEL_DIR / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"incomplete download: {name}")
    command = [
        str(ENGINE / "ds4-server"),
        "-m",
        str(MODEL_DIR / MODEL),
        "--ple",
        str(MODEL_DIR / PLE),
        "--metal",
        "--host",
        "127.0.0.1",
        "--port",
        "18136",
        "--ctx",
        "65536",
        "--power",
        "100",
        "--trace",
        str(OUT / f"{arm}-trace.log"),
    ]
    if arm == "ds4-mtp":
        command += ["--mtp", "--mtp-exact-sampling"]
    return command


def request(messages, model, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "think": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        bench.URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    text = []
    reasoning = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=660) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                if content:
                    first = first or time.perf_counter()
                    text.append(content)
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    count = usage.get("completion_tokens", 0)
    if not first or any(reasoning) or not count:
        raise RuntimeError(
            "invalid measurement: no visible output, hidden reasoning, or missing usage"
        )
    return {
        "ttft_seconds": first - start if first else None,
        "wall_seconds": end - start,
        "output_tokens_per_second": count / (end - start),
        "decode_tokens_per_second": (count - 1) / (end - first)
        if first and count > 1
        else None,
        "usage": usage,
        "content": "".join(text),
        "reasoning": "".join(reasoning),
        "finish_reason": finish,
        "messages": messages,
        "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
    }


def run(arm, smoke, *, workload=None, trace=True):
    command = server_command(arm)
    if not trace and "--trace" in command:
        trace_index = command.index("--trace")
        del command[trace_index : trace_index + 2]
    save(
        f"{arm}-method.json",
        {
            "command": command,
            "runtime": "oMLX 0.6.4" if arm == "baseline" else "DwarfStar",
            "candidate_engine_commit": bench.command(
                "git", "-C", str(ENGINE), "rev-parse", "HEAD"
            ).strip(),
            "candidate_artifact_revision": "59a55fb819c82be7b162948282b50bd1a1e290b7",
            "cold": "caller-defined workload; inspect per-request cached_tokens"
            if workload is not None
            else "request-specific prefix before shared body; resident model; cached_tokens must be zero",
            "warm": "caller-defined workload; inspect per-request cached_tokens"
            if workload is not None
            else "full conversation follow-up with identical history; observe actual cache behavior",
        },
    )
    save(f"{arm}-unloaded-before.json", bench.memory())
    log = (OUT / f"{arm}-console.log").open("w")
    proc = subprocess.Popen(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT)
    stop = threading.Event()

    def monitor():
        with (OUT / f"{arm}-memory.jsonl").open("w") as file:
            while not stop.is_set():
                file.write(json.dumps(bench.memory(proc.pid)) + "\n")
                file.flush()
                stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    model = "baseline" if arm == "baseline" else "qwen3.8-flash-next-chat"
    rows = []
    try:
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}; see log")
            if port_open(18136):
                break
            time.sleep(1)
        warm = request([{"role": "user", "content": "Return only READY."}], model, 16)
        save(f"{arm}-warmup.json", warm)
        print(arm, "warmup", warm["content"], flush=True)
        if bench.generic._normalized(warm["content"]) != "ready" or warm["reasoning"]:
            raise RuntimeError("warmup invalid or thinking not disabled")
        (OUT / f"{arm}-loaded-vmmap.txt").write_text(
            bench.command("vmmap", "-summary", str(proc.pid))
        )
        if smoke:
            return
        if workload is not None:
            workload(arm, model, proc.pid)
            save(f"{arm}-loaded-after.json", bench.memory(proc.pid))
            (OUT / f"{arm}-post-vmmap.txt").write_text(
                bench.command("vmmap", "-summary", str(proc.pid))
            )
            return
        for target in (4096, 16384, 32768):
            for repeat in range(1, 4):
                prompt, secret = bench.generic._needle_prompt(target, repeat)
                messages = [
                    {
                        "role": "user",
                        "content": f"Independent test {target}/{repeat}.\n" + prompt,
                    }
                ]
                result = request(messages, model, 16)
                result.update(
                    kind="cold",
                    target=target,
                    repeat=repeat,
                    passed=bench.generic._normalized(result["content"])
                    == secret.casefold(),
                )
                rows.append(result)
                save(f"{arm}-performance.json", rows)
                if (
                    result["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens")
                    != 0
                ):
                    raise RuntimeError(
                        "cold row cache miss is unverified or cached tokens were reused"
                    )
                print(
                    arm,
                    "cold",
                    target,
                    repeat,
                    round(result["ttft_seconds"] or 0, 3),
                    result["passed"],
                    result["usage"],
                    flush=True,
                )
                if target == 16384:
                    follow = messages + [
                        {"role": "assistant", "content": result["content"]},
                        {"role": "user", "content": "Repeat that secret value only."},
                    ]
                    result = request(follow, model, 16)
                    result.update(
                        kind="warm_followup",
                        target=target,
                        repeat=repeat,
                        passed=bench.generic._normalized(result["content"])
                        == secret.casefold(),
                    )
                    rows.append(result)
                    save(f"{arm}-performance.json", rows)
                    print(
                        arm,
                        "warm followup",
                        repeat,
                        result["ttft_seconds"],
                        result["usage"],
                        flush=True,
                    )
        for repeat in range(1, 4):
            prompt = f"試行{repeat}。日本語で、議事録から事実と推測を区別して要約する方法を具体例付きで詳しく説明してください。少なくとも800文字書いてください。"
            result = request([{"role": "user", "content": prompt}], model, 384)
            result.update(kind="decode", repeat=repeat)
            rows.append(result)
            save(f"{arm}-performance.json", rows)
            print(arm, "decode", repeat, result["decode_tokens_per_second"], flush=True)
        quality = []
        for ident, prompt, expected in (*bench.generic.EXACT_CASES, *bench.JP_CASES):
            result = request([{"role": "user", "content": prompt}], model, 128)
            result.update(
                id=ident,
                expected=expected,
                passed=bench.generic._normalized(result["content"])
                == bench.generic._normalized(expected),
            )
            quality.append(result)
            save(f"{arm}-quality.json", quality)
        print(
            arm,
            "quality",
            sum(r["passed"] for r in quality),
            "/",
            len(quality),
            flush=True,
        )
        save(f"{arm}-loaded-after.json", bench.memory(proc.pid))
        (OUT / f"{arm}-post-vmmap.txt").write_text(
            bench.command("vmmap", "-summary", str(proc.pid))
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        stop.set()
        thread.join(timeout=5)
        log.close()
        save(f"{arm}-unloaded-after.json", bench.memory())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["baseline", "ds4", "ds4-mtp", "self-check"])
    parser.add_argument("--download-pid", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.arm == "self-check":
        import io
        from unittest.mock import patch

        prompt, secret = bench.generic._needle_prompt(4096, 1)
        assert secret in prompt and len(bench.JP_CASES) == 12
        assert len(PROTECTED) == 3 and all(p.is_file() for p in PROTECTED)
        assert prompt_schema({"messages": []}) == {"messages": []}
        assert bench.memory()["os_including_inactive_bytes"] > 0
        events = [
            {"choices": [{"delta": {"content": "READY"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        ]
        wire = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        with patch.object(urllib.request, "urlopen", return_value=io.BytesIO(wire)):
            result = request([{"role": "user", "content": "test"}], "test", 8)
        assert result["content"] == "READY" and result["finish_reason"] == "stop"
        assert result["usage"]["completion_tokens"] == 2
        assert 0 <= result["ttft_seconds"] <= result["wall_seconds"]
        assert result["decode_tokens_per_second"] > 0 and not result["reasoning"]
        with (
            patch("__main__.port_open", side_effect=[True, False]),
            patch.object(
                subprocess, "check_output", side_effect=["123\n", "1 omlx-server\n"]
            ),
            patch("__main__.save"),
            patch.object(os, "kill") as kill,
        ):
            stop_orphan("test")
            kill.assert_called_once_with(123, signal.SIGTERM)
        with (
            patch("__main__.port_open", return_value=True),
            patch.object(
                subprocess, "check_output", side_effect=["123\n", "1 unrelated\n"]
            ),
            patch.object(os, "kill") as kill,
        ):
            try:
                stop_orphan("test")
            except RuntimeError:
                pass
            else:
                raise AssertionError("unexpected listeners must not be stopped")
            kill.assert_not_called()
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=io.BytesIO(b'data: {"error": "fixture error"}\n\n'),
        ):
            try:
                request([], "test", 8)
            except RuntimeError as exc:
                assert str(exc) == "fixture error"
            else:
                raise AssertionError("stream errors must fail the benchmark")
        print("self-check passed")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with isolated(args.arm, args.download_pid):
            run(args.arm, args.smoke)
