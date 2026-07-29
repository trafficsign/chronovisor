"""Debug helper: run triage + generate on a single raw file with full output capture.

Usage:
    uv run python scripts/debug_one_raw.py <raw-filename-or-path>

Wraps `chronovisor.ingest.ingest.generate` so every LLM call's full prompt + output
is dumped to /tmp/wiki-debug/<raw-stem>/. We then call `_triage` and
`_generate_one` directly and report which parser stage failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chronovisor.ingest.ingest as ingest
from chronovisor.ingest.ingest import (
    _extract_json_array,
    _extract_page_body,
    _generate_one,
    _triage,
    _validate_triage_plan,
)

WIKI_RAW = Path.home() / ".chronovisor" / "raw"
DEBUG_OUT = Path("/tmp/wiki-debug")


def resolve_raw(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    if (WIKI_RAW / arg).exists():
        return WIKI_RAW / arg
    matches = list(WIKI_RAW.glob(f"*{arg}*"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"raw not found: {arg}")
    raise SystemExit(f"ambiguous raw {arg!r}: {[m.name for m in matches]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--max-ops", type=int, default=None,
                    help="cap how many generate ops to run (default: all)")
    args = ap.parse_args()

    raw_path = resolve_raw(args.raw)
    out_dir = DEBUG_OUT / raw_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[debug] raw: {raw_path}")
    print(f"[debug] dump dir: {out_dir}")

    captures: list[tuple[str, str, str]] = []
    real_generate = ingest.generate
    call_idx = {"n": 0}

    def wrapped_generate(prompt: str, system: str | None = None) -> str:
        i = call_idx["n"]
        call_idx["n"] += 1
        out = real_generate(prompt, system=system)
        tag = f"call-{i:02d}"
        (out_dir / f"{tag}.system.txt").write_text(system or "")
        (out_dir / f"{tag}.prompt.txt").write_text(prompt)
        (out_dir / f"{tag}.output.txt").write_text(out)
        captures.append((tag, system or "", out))
        return out

    ingest.generate = wrapped_generate
    try:
        content = raw_path.read_text()
        print(f"[debug] raw size: {len(content)} chars")

        print("\n=== STAGE 1: TRIAGE ===")
        plan = _triage(content)
        triage_out = captures[0][2] if captures else ""
        print(f"[debug] triage output length: {len(triage_out)} chars")
        print(f"[debug] triage output last 200 chars: {triage_out[-200:]!r}")
        raw_plan = _extract_json_array(triage_out)
        if raw_plan is None:
            print("[FAIL] _extract_json_array returned None — could not find a JSON array")
        else:
            print(f"[debug] _extract_json_array → list[{len(raw_plan)}], "
                  f"first entry type: {type(raw_plan[0]).__name__ if raw_plan else 'empty'}")
            print(f"[debug] first entry: {raw_plan[0]!r}" if raw_plan else "")
            validated = _validate_triage_plan(raw_plan)
            if validated is None:
                print("[FAIL] _validate_triage_plan rejected the plan")
            else:
                print(f"[OK] triage valid: {len(validated)} ops")

        if plan is None:
            print("[result] triage failed — STOP")
            return 1
        if not plan:
            print("[result] triage produced empty plan — nothing to generate")
            return 0

        print(f"\n=== STAGE 2: GENERATE ({len(plan)} ops) ===")
        for i, op in enumerate(plan):
            if args.max_ops is not None and i >= args.max_ops:
                print(f"[debug] stopping at --max-ops={args.max_ops}")
                break
            fname = op.get("filename", "?")
            op_type = op.get("type", "?")
            before = call_idx["n"]
            print(f"\n--- op {i+1}/{len(plan)}: {op_type} {fname} ---")
            generated = _generate_one(op, content)
            after = call_idx["n"]
            for j in range(before, after):
                tag, _sys, out = captures[j]
                print(f"[debug] {tag} output length: {len(out)} chars")
                print(f"[debug] {tag} output head: {out[:160]!r}")
                print(f"[debug] {tag} output tail: {out[-200:]!r}")
                if generated is None:
                    body = _extract_page_body(out, op_type=op_type)
                    if body is None:
                        # Re-run each pattern to find which one failed
                        print(f"[FAIL] _extract_page_body returned None for {fname}")
                        import re
                        m1 = re.search(
                            r"===\s*(?:NEW|UPDATE)\s+PAGE:\s*\S+\s*===\n(.*?)\n===\s*END\s+PAGE\s*===",
                            out, re.DOTALL | re.IGNORECASE,
                        )
                        m2 = re.search(
                            r"===[^\n]*===\n(.*?)\n===\s*END\s+PAGE\s*===",
                            out, re.DOTALL | re.IGNORECASE,
                        )
                        starts_fence = out.lstrip().startswith("===")
                        has_end = "END PAGE" in out.upper()
                        starts_fm = out.lstrip().startswith("---")
                        print(f"        starts_fence={starts_fence} has_end={has_end} starts_fm={starts_fm}")
                        print(f"        strict pattern matched: {bool(m1)}")
                        print(f"        lenient pattern matched: {bool(m2)}")
            if generated is not None:
                print(f"[OK] generated {fname} (body {len(generated['content'])} chars)")
        return 0
    finally:
        ingest.generate = real_generate


if __name__ == "__main__":
    sys.exit(main())
