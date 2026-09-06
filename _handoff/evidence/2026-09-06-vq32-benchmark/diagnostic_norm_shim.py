"""Diagnostic only: trunk-only shim for mlx-lm 2196836, not the final runtime.

The final benchmark uses stock pre-fold mlx-lm 8a36d1e, without this shim.
No disk weights are edited. MTP norms are intentionally untouched here.
"""

import sys

import mlx.core as mx
from mlx_lm.models import qwen4_exp


def fold_flat_norms(weights):
    """The pinned artifact stores raw zero-centered norms under flat model.* keys."""
    if any(key.startswith("model.language_model.") for key in weights):
        raise ValueError("Raw HF layout already folds in upstream sanitize")
    return {
        key: value + 1.0 if key.endswith(qwen4_exp.Model._FOLD_ONE) else value
        for key, value in weights.items()
    }


def self_check():
    key = "model.layers.0.attn_hyper_connection.hc_norm.weight"
    gated = "model.layers.0.linear_attn.norm.weight"
    weights = {key: mx.array([0.25, -0.5]), gated: mx.array([0.9])}
    before = qwen4_exp.Model.sanitize(None, weights)
    assert bool(mx.array_equal(before[key], weights[key]))
    after = fold_flat_norms(before)
    assert bool(mx.array_equal(after[key], mx.array([1.25, 0.5])))
    assert after[gated] is weights[gated]
    assert bool(mx.array_equal(weights[key], mx.array([0.25, -0.5])))
    norm = qwen4_exp.RMSNorm(2)
    norm.weight = after[key]
    x = mx.array([[0.5, 2.0]])
    expected = mx.fast.rms_norm(x, None, norm.eps) * (1 + weights[key])
    assert bool(mx.allclose(norm(x), expected))
    print("zero-centered norm compatibility self-check: ok", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        original = qwen4_exp.Model.sanitize

        def sanitize(self, weights):
            result = fold_flat_norms(original(self, weights))
            count = sum(k.endswith(self._FOLD_ONE) for k in result)
            print(
                f"VQ benchmark shim: folded {count} zero-centered norm weights",
                flush=True,
            )
            return result

        qwen4_exp.Model.sanitize = sanitize
        from vqlab.serve import main

        main()
