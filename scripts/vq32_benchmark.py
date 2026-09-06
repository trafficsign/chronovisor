#!/usr/bin/env python3.14
"""VQLab benchmark using the existing paired tests; services managed separately."""

import argparse
import copy
import hashlib
import json
import socket
import urllib.request

import sawfwair_quant_benchmark as bench

WORK = bench.Path("/Users/trafficsign/.omlx/experiments/vq32")
REVISION = "b3a40c3590785c7276e6d51bda97486d0b179f0e"
MODEL_REPO = "TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw"


def prompt_schema(payload):
    """Same explicit schema prompt, without grammar decoding, for both engines."""
    payload = copy.deepcopy(payload)
    fmt = payload.pop("response_format", None)
    if fmt and fmt.get("type") == "json_schema":
        schema = fmt["json_schema"]["schema"]
        payload["messages"].append(
            {
                "role": "user",
                "content": "Return only a JSON object conforming to this output schema. "
                "Do not use Markdown or add explanations.\n"
                + json.dumps(schema, ensure_ascii=False, sort_keys=True),
            }
        )
    return payload


def verify_artifact():
    url = (
        f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{REVISION}?blobs=true"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    root = (WORK / "model").resolve()
    files = []
    for row in metadata["siblings"]:
        if row["rfilename"].endswith(".png"):
            continue
        path = (root / row["rfilename"]).resolve()
        assert path.is_relative_to(root) and path.is_file(), row["rfilename"]
        assert path.stat().st_size == row["size"], row["rfilename"]
        files.append({"file": row["rfilename"], "bytes": row["size"]})
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert all((root / shard).is_file() for shard in index["weight_map"].values())
    bench.save(
        "artifact.json",
        {
            "repo": MODEL_REPO,
            "revision": metadata["sha"],
            "files": files,
            "model_py_sha256": hashlib.sha256(
                (root / "model.py").read_bytes()
            ).hexdigest(),
            "runtime_commit": bench.command(
                "git", "-C", str(WORK / "VQLab"), "rev-parse", "HEAD"
            ).strip(),
        },
    )


def run(arm, smoke_only=False):
    for port in (18125, 18136):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        except OSError:
            continue
        raise RuntimeError(f"Port {port} still open; refusing overlapping benchmark")
    bench.WORK = WORK
    bench.OUT = bench.ROOT / "_handoff/evidence/2026-09-06-vq32-benchmark"
    bench.OUT.mkdir(parents=True, exist_ok=True)
    candidate = arm != "baseline"
    if candidate:
        verify_artifact()
    real_post = bench.generic._post
    real_quality = bench.generic._quality
    real_stream = bench.stream

    def post(url, payload, timeout):
        payload = prompt_schema(payload)
        if candidate:
            payload["model"] = "default_model"
        result = real_post(url, payload, timeout)
        with (bench.OUT / f"{arm}-responses.jsonl").open("a") as file:
            file.write(
                json.dumps({"request": payload, "response": result}, ensure_ascii=False)
                + "\n"
            )
        return result

    def quality(url, model, timeout):
        schema = {
            "type": "object",
            "properties": {"probe": {"type": "string", "enum": ["NATIVE_SCHEMA"]}},
            "required": ["probe"],
            "additionalProperties": False,
        }
        result = real_post(
            url,
            {
                "model": "default_model" if candidate else model,
                "messages": [{"role": "user", "content": "Return a JSON object."}],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "probe", "strict": True, "schema": schema},
                },
            },
            timeout,
        )
        bench.save(f"{arm}-native-schema-probe.json", result)
        return real_quality(url, model, timeout)

    bench.generic._post = post
    bench.generic._quality = quality
    bench.corpus.post = post
    bench.stream = lambda prompt, model, max_tokens=256: real_stream(
        prompt, "default_model" if candidate else model, max_tokens
    )
    command = None
    if candidate:
        command = [
            str(WORK / "venv/bin/python"),
            "-u",
            "-m",
            "vqlab.cli",
            "serve",
            "--model",
            str(WORK / "model"),
            "--host",
            "127.0.0.1",
            "--port",
            "18136",
            "--prompt-cache-size",
            "0",
            "--decode-concurrency",
            "1",
        ]
        if arm == "vq-normfix":
            command[2:5] = [str(bench.OUT / "diagnostic_norm_shim.py")]
        if arm in {"vq-mtp", "vq-normfix", "vq-compatible"}:
            command += ["--sidecar", str(WORK / "model/mtp-head-q6.safetensors")]
    bench.save(
        f"{arm}-method.json",
        {
            "arm": arm,
            "command": command,
            "schema_mode": "identical explicit prompt; response_format removed for paired quality tests",
            "prefix_cache": False,
        },
    )
    bench.benchmark(arm, server_command=command, smoke_only=smoke_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arm",
        choices=[
            "baseline",
            "vq-mtp",
            "candidate-no-mtp",
            "vq-prefill-only",
            "vq-normfix",
            "vq-compatible",
            "vq-compatible-no-mtp",
            "self-check",
        ],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.arm == "self-check":
        payload = {
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
        }
        changed = prompt_schema(payload)
        assert "response_format" in payload and "response_format" not in changed
        assert len(payload["messages"]) == 1 and len(changed["messages"]) == 2
        assert prompt_schema({"messages": []}) == {"messages": []}
        bench.corpus.self_check()
        print("self-check: ok")
    else:
        run(args.arm, smoke_only=args.smoke)
