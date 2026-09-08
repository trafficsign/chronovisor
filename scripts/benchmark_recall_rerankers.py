"""Compare local rerankers on a frozen synthetic Japanese candidate set.

This is a diagnostic of selection, not a production Recall acceptance test.
It never searches or modifies the live memory store. Download pinned model
snapshots separately and pass their manifest with --models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPOS = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "mem": "nisavid/MemReranker-4B-OptiQ-4bit",
    "lychee": "fuhao23/reranker_v1",
    "japanese": "hotchpotch/japanese-reranker-xsmall-v2",
}
SEED = 20260908
MAX_LENGTH = 512


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    assert data["kind"] == "synthetic_diagnostic"
    cases = data["cases"]
    assert cases and len({c["id"] for c in cases}) == len(cases)
    rng = random.Random(SEED)
    for case in cases:
        candidates = case["candidates"]
        assert len(candidates) >= 3
        assert len({c["id"] for c in candidates}) == len(candidates)
        assert all(
            c["text"].strip() and c["relevance"] in (0, 1, 2) for c in candidates
        )
        assert (case["category"] == "no_evidence") == (
            not any(c["relevance"] for c in candidates)
        )
        rng.shuffle(candidates)
    return cases


def query_text(case: dict, *, with_context: bool = True) -> str:
    context = case.get("context", "")
    if isinstance(context, list):
        context = "\n".join(context)
    return (
        f"直前の会話:\n{context}\n\n今回の質問:\n{case['query']}"
        if with_context and context
        else case["query"]
    )


def ranking_metrics(candidates: list[dict], scores: list[float]) -> dict:
    if len(scores) != len(candidates) or not all(math.isfinite(v) for v in scores):
        raise ValueError("Scores must be finite and match the candidate count")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    positives = {i for i, c in enumerate(candidates) if c["relevance"] > 0}
    result = {
        "ranking": [candidates[i]["id"] for i in order],
        "top_score": scores[order[0]],
        "has_evidence": bool(positives),
    }
    if not positives:
        return result
    ranks = [rank + 1 for rank, i in enumerate(order) if i in positives]
    grades = [c["relevance"] for c in candidates]
    dcg = sum(
        (2 ** grades[i] - 1) / math.log2(rank + 2) for rank, i in enumerate(order[:3])
    )
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(sorted(grades, reverse=True)[:3])
    )
    result.update(
        top1=int(order[0] in positives),
        mrr=1 / min(ranks),
        recall_at3=sum(r <= 3 for r in ranks) / len(ranks),
        all_evidence_at3=int(max(ranks) <= 3),
        ndcg_at3=dcg / ideal,
    )
    return result


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def make_scorer(key: str, snapshots: dict[str, str]):
    """Return score(query, texts), token_lengths(query, texts), metadata."""
    path = snapshots[REPOS[key]]
    meta = {"max_length": MAX_LENGTH, "batch_size": 6}
    if key in {"bge", "japanese"}:
        import torch

        from chronovisor.core.reranker import _MODEL_CACHE, _transformer_scores
        from chronovisor.core.runtime_config import RerankerConfig

        if not torch.backends.mps.is_available():
            raise RuntimeError("This comparison requires the Apple MPS device")
        torch.set_num_threads(4)
        config = RerankerConfig(
            enabled=True,
            model=path,
            device="mps",
            dtype="float32",
            max_length=MAX_LENGTH,
            batch_size=6,
        )
        start = time.perf_counter()
        _transformer_scores("初期化", ["モデルの初期化です。"], config)
        tokenizer, _model = next(iter(_MODEL_CACHE.values()))
        meta.update(
            backend="production_transformer_scorer",
            device="mps",
            dtype="float32",
            load_and_first_score_s=time.perf_counter() - start,
        )

        def score(query, texts):
            values = _transformer_scores(query, texts, config)
            torch.mps.synchronize()
            return values

        def lengths(query, texts):
            return [
                len(row)
                for row in tokenizer([query] * len(texts), texts, truncation=False)[
                    "input_ids"
                ]
            ]

        return score, lengths, meta

    if key == "lychee":
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        base_path = snapshots["Qwen/Qwen3-Reranker-0.6B"]
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base, loading = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=1,
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, path, is_trainable=False)
        expected = load_file(str(Path(path) / "adapter_model.safetensors"))[
            "base_model.model.score.weight"
        ].float()
        heads = [
            (n, v)
            for n, v in model.named_parameters()
            if "score.modules_to_save.default.weight" in n
        ]
        if len(heads) != 1 or not torch.equal(heads[0][1].detach().float(), expected):
            raise RuntimeError(
                "The trained Lychee classification head was not restored"
            )
        model.to(device="mps", dtype=torch.float32).eval()
        torch.mps.synchronize()
        meta.update(
            backend="peft_sequence_classifier",
            device="mps",
            dtype="float32",
            trained_head_verified=True,
            base_loading_info=loading,
            load_s=time.perf_counter() - start,
        )

        def pairs(query, texts):
            return [
                "<Instruct>: Given a user query, retrieve memory snippets that "
                f"answer the query\n<Query>: {query}\n<Document>: {text}"
                for text in texts
            ]

        def score(query, texts):
            inputs = tokenizer(
                pairs(query, texts),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
            if list(logits.shape) != [len(texts), 1]:
                raise ValueError(f"Unexpected classifier shape: {logits.shape}")
            values = logits[:, 0].float().cpu().tolist()
            torch.mps.synchronize()
            return values

        def lengths(query, texts):
            return [
                len(row)
                for row in tokenizer(pairs(query, texts), truncation=False)["input_ids"]
            ]

        return score, lengths, meta

    import mlx.core as mx
    from mlx_lm import load

    start = time.perf_counter()
    config = json.loads((Path(path) / "config.json").read_text())
    # The published conversion uses Transformers 5's nested RoPE configuration.
    # MLX LM 0.31.3 requires the equivalent flat key; leave the snapshot intact.
    rope = config.get("rope_parameters", {})
    if rope.get("rope_type") != "default":
        raise ValueError("Only the published default RoPE configuration is supported")
    override = {"rope_theta": rope["rope_theta"]}
    model, tokenizer = load(path, model_config=override)
    mx.eval(model.parameters())
    meta.update(
        backend="mlx_qwen3_yes_no",
        device="gpu",
        dtype="published_mixed_4_8bit",
        load_s=time.perf_counter() - start,
        in_memory_config_override=override,
    )
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer "
        'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre = tokenizer.encode(prefix, add_special_tokens=False)
    post = tokenizer.encode(suffix, add_special_tokens=False)
    no_id, yes_id = [tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def tokens(query, texts, *, truncate=True):
        result = []
        for text in texts:
            body = tokenizer.encode(
                "<Instruct>: Given a user query, retrieve memory snippets that "
                f"answer the query\n<Query>: {query}\n<Document>: {text}",
                add_special_tokens=False,
            )
            if truncate:
                body = body[: MAX_LENGTH - len(pre) - len(post)]
            result.append(pre + body + post)
        return result

    def score(query, texts):
        rows = tokens(query, texts)
        width = max(map(len, rows))
        ids = mx.array([[pad_id] * (width - len(row)) + row for row in rows])
        # Explicit padding mask: MLX's plain Qwen3 call only makes a causal mask.
        positions = mx.arange(width)
        valid_keys = (
            positions[None, :] >= mx.array([width - len(r) for r in rows])[:, None]
        )
        causal = positions[:, None] >= positions[None, :]
        mask = (causal[None, None, :, :] & valid_keys[:, None, None, :]) | mx.eye(
            width, dtype=mx.bool_
        )[None, None, :, :]
        core = model.model
        h = core.embed_tokens(ids)
        for layer in core.layers:
            h = layer(h, mask=mask)
        h = core.norm(h[:, -1:, :])
        logits = (
            core.embed_tokens.as_linear(h)
            if model.args.tie_word_embeddings
            else model.lm_head(h)
        )[:, 0, :]
        values = mx.softmax(logits[:, [no_id, yes_id]].astype(mx.float32), axis=-1)[
            :, 1
        ]
        mx.eval(values)
        return values.tolist()

    def lengths(query, texts):
        return [len(row) for row in tokens(query, texts, truncate=False)]

    # Check the optimized batched last-token path against stock unpadded calls.
    q = "どの色を選んだ？"
    docs = ["青に決めた。", "長い説明だが、これは録音設定を変更した記録です。"]
    batched = score(q, docs)
    reference = []
    for row in tokens(q, docs):
        logits = model(mx.array([row]))[:, -1, :].astype(mx.float32)
        probability = mx.softmax(logits[:, [no_id, yes_id]], axis=-1)[:, 1]
        mx.eval(probability)
        reference.extend(probability.tolist())
    delta = max(abs(a - b) for a, b in zip(batched, reference, strict=True))
    meta["batch_vs_stock_max_abs_probability_error"] = delta
    if delta > 0.01:
        raise RuntimeError(f"MLX batching disagrees with stock scoring: {delta}")
    return score, lengths, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REPOS, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "tests/fixtures/recall_reranker_japanese.json",
    )
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    manifest = json.loads(args.models.read_text())
    cases = load_cases(args.cases)
    result = {
        "model": args.model,
        "repo": REPOS[args.model],
        "status": "running",
        "dataset_kind": "synthetic_diagnostic",
        "seed": SEED,
        "fixture_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True
        ).strip(),
        "versions": {},
        "rows": [],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for package in ("torch", "transformers", "mlx", "mlx-lm", "peft", "safetensors"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.perf_counter()
    try:
        score, lengths, meta = make_scorer(
            args.model, {m["repo"]: m["path"] for m in manifest}
        )
        result["runtime"] = meta
        warm = cases[0]
        warm_texts = [c["text"] for c in warm["candidates"]]
        warm_times = []
        for _ in range(2):
            t = time.perf_counter()
            score(query_text(warm), warm_texts)
            warm_times.append((time.perf_counter() - t) * 1000)
        result["warmup_ms"] = warm_times
        for repeat in range(args.repeats):
            for case in cases:
                query = query_text(case)
                texts = [c["text"] for c in case["candidates"]]
                token_lengths = lengths(query, texts)
                t = time.perf_counter()
                scores = score(query, texts)
                elapsed = (time.perf_counter() - t) * 1000
                row = {
                    "id": case["id"],
                    "category": case["category"],
                    "repeat": repeat,
                    "mode": "with_context",
                    "latency_ms": elapsed,
                    "scores": scores,
                    "candidate_ids": [c["id"] for c in case["candidates"]],
                    "untruncated_token_lengths": token_lengths,
                    "truncated_pairs": sum(n > MAX_LENGTH for n in token_lengths),
                    **ranking_metrics(case["candidates"], scores),
                }
                result["rows"].append(row)
                print(
                    json.dumps(
                        {k: row.get(k) for k in ("id", "repeat", "top1", "latency_ms")}
                    ),
                    flush=True,
                )
        # Bounded context ablation, kept out of primary timing/quality aggregates.
        for case in cases:
            if case["category"] not in {"coreference", "topic_shift"}:
                continue
            t = time.perf_counter()
            scores = score(
                query_text(case, with_context=False),
                [c["text"] for c in case["candidates"]],
            )
            result["rows"].append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "mode": "query_only",
                    "repeat": 0,
                    "latency_ms": (time.perf_counter() - t) * 1000,
                    "scores": scores,
                    **ranking_metrics(case["candidates"], scores),
                }
            )
        # A separate latency-only workload; repeated filler is not a quality test.
        long_query = "前回の検索方式と、変更後の評価方法を確認したい。"
        long_texts = [
            f"比較資料{index}。"
            + "検索候補の内容と変更履歴を保存し、質問に必要な情報を確認する。" * 40
            for index in range(6)
        ]
        long_lengths = lengths(long_query, long_texts)
        if min(long_lengths) < MAX_LENGTH:
            raise ValueError("The long workload must fill the configured token window")
        for _ in range(2):
            score(long_query, long_texts)
        long_timings = []
        for _ in range(3):
            t = time.perf_counter()
            long_scores = score(long_query, long_texts)
            if len(long_scores) != 6 or not all(math.isfinite(s) for s in long_scores):
                raise ValueError("Non-finite long-workload scores")
            long_timings.append((time.perf_counter() - t) * 1000)
        result["long_workload"] = {
            "kind": "latency_only_repeated_filler_not_quality",
            "candidates": 6,
            "tokens_per_pair_after_truncation": MAX_LENGTH,
            "untruncated_token_lengths": long_lengths,
            "warmup_queries": 2,
            "timings_ms": long_timings,
            "p50_ms": statistics.median(long_timings),
        }
        measured = [r for r in result["rows"] if r["mode"] == "with_context"]
        positive = [r for r in measured if r["repeat"] == 0 and r["has_evidence"]]
        latencies = [r["latency_ms"] for r in measured]
        result["summary"] = {
            "positive_cases": len(positive),
            "timed_queries": len(latencies),
            "top1_correct": sum(r["top1"] for r in positive),
            **{
                metric: statistics.mean(r[metric] for r in positive)
                for metric in (
                    "top1",
                    "mrr",
                    "recall_at3",
                    "all_evidence_at3",
                    "ndcg_at3",
                )
            },
            "p50_ms": statistics.median(latencies),
            "p95_ms": percentile(latencies, 0.95),
            "truncated_pairs": sum(r["truncated_pairs"] for r in measured),
            "no_evidence_top_scores": [
                r["top_score"]
                for r in measured
                if r["repeat"] == 0 and not r["has_evidence"]
            ],
            "abstention_evaluated": False,
        }
        result["status"] = "complete"
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.perf_counter() - started
        result["peak_process_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "summary": result.get("summary"),
                    "error": result.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
