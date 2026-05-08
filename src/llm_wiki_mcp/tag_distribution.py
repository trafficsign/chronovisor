"""Tag distribution report (plan-3).

Read-only analysis. Samples 200 pages from the corpus, asks the LLM to
tag each one twice — once restricted to the v0.1 master list
(``assigned_tags``), once unrestricted (``suggested_missing_categories``)
— and emits an aggregate report so we can decide whether the taxonomy
is fit for purpose before rolling tag generation out to all 1631 pages.

Direction of inference goes one way only: the script reads pages and
writes an external report. Pages on disk are NEVER modified.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from llm_wiki_mcp.tags import AXIS_LIMITS, SEED_TAGS, parse_tags, validate_axis_counts
from llm_wiki_mcp.wiki import find_page


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SamplingPlan:
    page_ids: list[str]
    weights: dict[str, float]
    folder_distribution: dict[str, int]
    seed: int


@dataclass
class PageAnalysis:
    page_id: str
    assigned_tags: list[str] = field(default_factory=list)
    rejected_assigned_tags: list[str] = field(default_factory=list)
    suggested_missing_categories: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""  # for the audit log


# ---------------------------------------------------------------------------
# LLM contract
# ---------------------------------------------------------------------------


def _flatten_master(master: dict[str, list[str]]) -> list[str]:
    return [t for axis in master.values() for t in axis]


TAG_REPORT_SYSTEM_PROMPT = """\
You are auditing whether a controlled tag taxonomy fits a real wiki page.

You will be given:
  - a TAG MASTER LIST (the only tags that may appear in ``assigned_tags``)
  - a PAGE (title + body head)

Output ONE JSON object with exactly these fields:

{
  "assigned_tags": ["d/...", "t/...", "s/..."],
  "rejected_assigned_tags": ["d/automotive"],
  "suggested_missing_categories": [
    {"label": "automotive-engineering", "justification": "<one sentence>", "fallback_axis": "d/"}
  ],
  "confidence": 0.0
}

Rules:
- ``assigned_tags``: ONLY values from the master list. Each entry must
  appear verbatim in the master. Aim for the canonical set: 1-3 d/, 1 t/,
  1 s/. Pick fewer if no master tag fits — empty list is acceptable.
- ``rejected_assigned_tags``: tags you wanted to assign but had to drop
  because they aren't on the master list. This is the hallucination
  audit channel.
- ``suggested_missing_categories``: free-form. Each item is an object
  with ``label`` (kebab-case body without prefix), ``justification`` (one
  short sentence in Japanese), and ``fallback_axis`` (one of "d/", "t/",
  "s/" — best guess). Empty list is fine if the master suffices.
- ``confidence``: float in [0, 1] reflecting how well the master
  taxonomy describes the page. Low confidence = the page sits awkwardly
  in this taxonomy.
- Output JSON ONLY, no preamble, no markdown fences.
"""


_REQUIRED_FIELDS = {
    "assigned_tags",
    "rejected_assigned_tags",
    "suggested_missing_categories",
    "confidence",
}


def parse_llm_response(raw: str, master_set: set[str]) -> dict | None:
    """Validate the LLM JSON output. Returns ``None`` on any breach so
    the caller can keep the raw text in the audit log without fabricating
    structured fields."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) - _REQUIRED_FIELDS:
        return None
    if not _REQUIRED_FIELDS.issubset(obj.keys()):
        return None

    # assigned_tags: list[str], split into in-master / out-of-master.
    raw_assigned = obj.get("assigned_tags", [])
    if not isinstance(raw_assigned, list) or not all(
        isinstance(t, str) for t in raw_assigned
    ):
        return None
    in_master = [t for t in raw_assigned if t in master_set]
    leaked = [t for t in raw_assigned if t not in master_set]

    # rejected_assigned_tags: list[str] (the LLM's own audit channel).
    rejected = obj.get("rejected_assigned_tags", [])
    if not isinstance(rejected, list) or not all(isinstance(t, str) for t in rejected):
        return None

    # suggested_missing_categories: list[dict] with strict shape.
    smc_raw = obj.get("suggested_missing_categories", [])
    if not isinstance(smc_raw, list):
        return None
    smc: list[dict] = []
    for item in smc_raw:
        if not isinstance(item, dict):
            return None
        if set(item.keys()) - {"label", "justification", "fallback_axis"}:
            return None
        if not {"label", "justification", "fallback_axis"}.issubset(item.keys()):
            return None
        if not all(
            isinstance(item[k], str)
            for k in ("label", "justification", "fallback_axis")
        ):
            return None
        smc.append(
            {
                "label": item["label"].strip(),
                "justification": item["justification"].strip(),
                "fallback_axis": item["fallback_axis"].strip(),
            }
        )

    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)):
        return None
    if not (0.0 <= float(confidence) <= 1.0):
        return None

    return {
        "assigned_tags": in_master,
        # Merge the LLM's self-flagged rejections with anything we caught
        # leaking past the master gate. Keep duplicates de-duped.
        "rejected_assigned_tags": sorted(set(rejected) | set(leaked)),
        "suggested_missing_categories": smc,
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _folder_of(page_id: str, store) -> str:
    """Folder segment from the stored absolute path. ``""`` for root pages."""
    meta = store.meta(page_id) or {}
    path = meta.get("path", "")
    if not path:
        return ""
    parts = Path(path).parts
    # PAGES_DIR is one of the parents; the folder is everything between
    # PAGES_DIR and the file, joined.
    try:
        from llm_wiki_mcp.wiki import PAGES_DIR
        rel = Path(path).relative_to(PAGES_DIR)
        if len(rel.parts) <= 1:
            return ""
        return rel.parts[0]
    except Exception:
        # Fall back to the immediate parent dir name.
        return parts[-2] if len(parts) >= 2 else ""


def proportional_sample(
    page_ids: list[str],
    store,
    n: int,
    seed: int,
) -> SamplingPlan:
    """Sample ``n`` page IDs proportional to folder population.

    Folder weights are recorded so downstream aggregation can up-weight
    rare folders in master-list frequency stats. Seed makes the draw
    reproducible across sessions.
    """
    rng = random.Random(seed)

    by_folder: dict[str, list[str]] = {}
    for pid in page_ids:
        folder = _folder_of(pid, store)
        by_folder.setdefault(folder, []).append(pid)

    total = len(page_ids)
    folder_dist = {f: len(ids) for f, ids in by_folder.items()}

    # Largest-remainder allocation: take floor of fractional quotas, then
    # distribute the leftover slots to folders with the largest fractional
    # parts. Avoids systematic underrepresentation of small folders.
    quotas: list[tuple[str, float, int]] = []
    for folder, ids in by_folder.items():
        fractional = (len(ids) / total) * n
        floor = int(fractional)
        quotas.append((folder, fractional - floor, floor))

    allocated = sum(q[2] for q in quotas)
    leftover = n - allocated
    quotas.sort(key=lambda q: -q[1])
    final: dict[str, int] = {}
    for i, (folder, _frac, base) in enumerate(quotas):
        bonus = 1 if i < leftover else 0
        cap = len(by_folder[folder])
        final[folder] = min(cap, base + bonus)

    drawn: list[str] = []
    for folder, take in final.items():
        if take == 0:
            continue
        drawn.extend(rng.sample(by_folder[folder], take))

    weights = {pid: len(by_folder[_folder_of(pid, store)]) / total for pid in drawn}

    return SamplingPlan(
        page_ids=sorted(drawn),
        weights=weights,
        folder_distribution=folder_dist,
        seed=seed,
    )


def minority_sample(
    page_ids: list[str],
    store,
    n: int,
    seed: int,
    dominant_threshold: float = 0.5,
) -> SamplingPlan:
    """Sample ``n`` pages from non-dominant folders, evenly across folders.

    A folder is "dominant" if it owns more than ``dominant_threshold`` of
    the corpus. Excluding it lets the LLM exercise the long-tail content
    that proportional sampling would miss almost entirely.
    """
    rng = random.Random(seed + 1)  # different draw than proportional

    by_folder: dict[str, list[str]] = {}
    for pid in page_ids:
        folder = _folder_of(pid, store)
        by_folder.setdefault(folder, []).append(pid)

    total = len(page_ids)
    minority = {
        f: ids
        for f, ids in by_folder.items()
        if len(ids) / total <= dominant_threshold
    }
    if not minority:
        return SamplingPlan(
            page_ids=[], weights={}, folder_distribution={}, seed=seed + 1
        )

    folders = sorted(minority.keys())
    per_folder = max(1, n // len(folders))

    drawn: list[str] = []
    for f in folders:
        ids = minority[f]
        take = min(per_folder, len(ids))
        drawn.extend(rng.sample(ids, take))

    # Trim or pad to reach exactly n, drawing remaining from the same pool.
    if len(drawn) > n:
        drawn = rng.sample(drawn, n)
    elif len(drawn) < n:
        remaining = [pid for ids in minority.values() for pid in ids if pid not in set(drawn)]
        rng.shuffle(remaining)
        drawn.extend(remaining[: n - len(drawn)])

    weights = {pid: 1.0 for pid in drawn}  # unweighted for taxonomy-gap analysis

    return SamplingPlan(
        page_ids=sorted(drawn),
        weights=weights,
        folder_distribution={f: len(ids) for f, ids in minority.items()},
        seed=seed + 1,
    )


# ---------------------------------------------------------------------------
# LLM scoring
# ---------------------------------------------------------------------------


def _page_head_for_prompt(page_id: str, max_chars: int = 2000) -> tuple[str, str]:
    """Return ``(title, body_head)``. Empty strings when the page is gone."""
    path = find_page(page_id)
    if path is None:
        return "", ""
    try:
        text = path.read_text()
    except OSError:
        return "", ""
    title_match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else page_id
    body = re.sub(r"^---\n.*?\n---\n?", "", text, count=1, flags=re.DOTALL).lstrip()
    return title, body[:max_chars]


def _build_prompt(page_id: str, master: list[str]) -> str:
    title, body = _page_head_for_prompt(page_id)
    master_str = "\n".join(f"  {t}" for t in master)
    return f"""\
TAG MASTER LIST:
{master_str}

PAGE:
  page_id: {page_id}
  title: {title}
  body head:
{body}

Output the JSON object per the rules.
"""


def analyze_page(
    page_id: str,
    master: list[str],
    generate_fn: Callable[..., str],
) -> PageAnalysis:
    """Tag one page. Returns a populated PageAnalysis even on LLM failure
    so the report can show "no result" cells with the raw text recorded
    for audit."""
    prompt = _build_prompt(page_id, master)
    try:
        raw = generate_fn(prompt, system=TAG_REPORT_SYSTEM_PROMPT)
    except Exception:
        return PageAnalysis(page_id=page_id, raw_response="<ollama error>")

    parsed = parse_llm_response(raw, set(master))
    if parsed is None:
        return PageAnalysis(page_id=page_id, raw_response=raw)

    return PageAnalysis(
        page_id=page_id,
        assigned_tags=parsed["assigned_tags"],
        rejected_assigned_tags=parsed["rejected_assigned_tags"],
        suggested_missing_categories=parsed["suggested_missing_categories"],
        confidence=parsed["confidence"],
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    plan_a: SamplingPlan,
    plan_b: SamplingPlan,
    results_a: list[PageAnalysis],
    results_b: list[PageAnalysis],
) -> dict:
    """Reduce per-page analyses to the report's headline stats."""
    # Master frequency, weighted (sample A) and unweighted (sample B).
    weighted: Counter = Counter()
    for r in results_a:
        w = plan_a.weights.get(r.page_id, 1.0)
        for tag in r.assigned_tags:
            weighted[tag] += w
    unweighted_b: Counter = Counter()
    for r in results_b:
        for tag in r.assigned_tags:
            unweighted_b[tag] += 1

    # Axis count violations (using validated tag set per page).
    violations: Counter = Counter()
    for r in results_a + results_b:
        parsed = parse_tags(r.assigned_tags)
        msgs = validate_axis_counts(parsed)
        for m in msgs:
            for axis in AXIS_LIMITS:
                if axis in m:
                    violations[axis] += 1
                    break
            else:
                violations["other"] += 1

    # Hallucination: anything that landed in ``rejected_assigned_tags``.
    hallucination_pages = sum(
        1 for r in results_a + results_b if r.rejected_assigned_tags
    )

    # Taxonomy gap: top labels from suggested_missing_categories.
    gap_counter: Counter = Counter()
    for r in results_a + results_b:
        for item in r.suggested_missing_categories:
            label = item.get("label", "").lower()
            if label:
                gap_counter[label] += 1

    # Empty assigned_tags rate.
    no_assign_a = sum(1 for r in results_a if not r.assigned_tags)
    no_assign_b = sum(1 for r in results_b if not r.assigned_tags)

    confidences = [r.confidence for r in results_a + results_b if r.raw_response]

    return {
        "weighted_master_frequency": dict(weighted.most_common()),
        "unweighted_master_frequency_b": dict(unweighted_b.most_common()),
        "axis_violations": dict(violations),
        "hallucination_pages": hallucination_pages,
        "hallucination_rate": (
            hallucination_pages / max(1, len(results_a) + len(results_b))
        ),
        "taxonomy_gap_top30": gap_counter.most_common(30),
        "empty_assigned_a": no_assign_a,
        "empty_assigned_b": no_assign_b,
        "confidence_min": min(confidences) if confidences else 0.0,
        "confidence_avg": sum(confidences) / len(confidences) if confidences else 0.0,
        "confidence_max": max(confidences) if confidences else 0.0,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    plan_a: SamplingPlan,
    plan_b: SamplingPlan,
    results_a: list[PageAnalysis],
    results_b: list[PageAnalysis],
    stats: dict,
    *,
    population_total: int,
    elapsed: float,
) -> str:
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append("---")
    lines.append("title: Tag Distribution Report (dry-run)")
    lines.append(f"updated: {today}")
    lines.append("---")
    lines.append("")
    lines.append("# Tag Distribution Report (dry-run)")
    lines.append("")
    lines.append("Read-only audit. Pages on disk were NOT modified.")
    lines.append("")

    lines.append("## Sampling")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- elapsed_seconds: {elapsed:.1f}")
    lines.append(f"- population_total: {population_total}")
    lines.append(
        f"- sample A (proportional): N={len(plan_a.page_ids)}, seed={plan_a.seed}"
    )
    lines.append(
        f"- sample B (minority oversample): N={len(plan_b.page_ids)}, seed={plan_b.seed}"
    )
    lines.append("")

    lines.append("## 1. Master list frequency")
    lines.append("### Sample A (weighted)")
    if not stats["weighted_master_frequency"]:
        lines.append("- _no master tags assigned_")
    else:
        for tag, freq in stats["weighted_master_frequency"].items():
            lines.append(f"- `{tag}`: {freq:.1f}")
    lines.append("")
    lines.append("### Sample B (unweighted, taxonomy-gap focus)")
    if not stats["unweighted_master_frequency_b"]:
        lines.append("- _no master tags assigned_")
    else:
        for tag, freq in stats["unweighted_master_frequency_b"].items():
            lines.append(f"- `{tag}`: {freq}")
    lines.append("")

    lines.append("## 2. Axis violations")
    if not stats["axis_violations"]:
        lines.append("- _none recorded_")
    else:
        for axis, count in stats["axis_violations"].items():
            lines.append(f"- `{axis}`: {count} page(s)")
    lines.append(
        f"- empty assigned_tags: A={stats['empty_assigned_a']}, B={stats['empty_assigned_b']}"
    )
    lines.append("")

    lines.append("## 3. Taxonomy gap (top 30 suggested_missing_categories)")
    if not stats["taxonomy_gap_top30"]:
        lines.append("- _no missing categories surfaced_")
    else:
        for label, count in stats["taxonomy_gap_top30"]:
            lines.append(f"- `{label}`: {count}")
    lines.append("")

    lines.append("## 4. Hallucination (master-list violations)")
    lines.append(
        f"- pages with rejected_assigned_tags: {stats['hallucination_pages']} "
        f"(rate: {stats['hallucination_rate']:.2%})"
    )
    lines.append("")

    lines.append("## 5. Confidence distribution")
    lines.append(
        f"- min={stats['confidence_min']:.2f} "
        f"avg={stats['confidence_avg']:.2f} "
        f"max={stats['confidence_max']:.2f}"
    )
    lines.append("")

    lines.append("## Appendix A: extracted page_id list")
    lines.append("### sample A")
    for pid in plan_a.page_ids:
        lines.append(f"- {pid}")
    lines.append("### sample B")
    for pid in plan_b.page_ids:
        lines.append(f"- {pid}")
    lines.append("")

    lines.append("## Appendix B: folder distribution")
    lines.append("### sample A folder distribution (population)")
    for folder, count in sorted(plan_a.folder_distribution.items()):
        lines.append(f"- `{folder or '(root)'}`: {count}")
    lines.append("### sample B folder distribution (minority pool)")
    for folder, count in sorted(plan_b.folder_distribution.items()):
        lines.append(f"- `{folder or '(root)'}`: {count}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Top-level dry-run runner
# ---------------------------------------------------------------------------


def run_dry_run(
    output_path: Path,
    raw_log_path: Path,
    *,
    sample_a_n: int = 100,
    sample_b_n: int = 100,
    seed: int = 42,
    store=None,
    generate_fn: Callable[..., str] | None = None,
) -> dict:
    if store is None:
        from llm_wiki_mcp.index_store import get_store
        store = get_store()
        store.refresh()
    if generate_fn is None:
        from llm_wiki_mcp.ollama import generate
        generate_fn = generate

    page_ids = sorted(store.all_page_ids(include_system=False))
    population_total = len(page_ids)

    plan_a = proportional_sample(page_ids, store, sample_a_n, seed=seed)
    plan_b = minority_sample(page_ids, store, sample_b_n, seed=seed)

    master = _flatten_master(SEED_TAGS)
    started = datetime.now()

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]

    results_a: list[PageAnalysis] = []
    for pid in plan_a.page_ids:
        analysis = analyze_page(pid, master, generate_fn)
        results_a.append(analysis)
        _append_raw_log(raw_log_path, run_id, "A", analysis)

    results_b: list[PageAnalysis] = []
    for pid in plan_b.page_ids:
        analysis = analyze_page(pid, master, generate_fn)
        results_b.append(analysis)
        _append_raw_log(raw_log_path, run_id, "B", analysis)

    stats = aggregate(plan_a, plan_b, results_a, results_b)
    elapsed = (datetime.now() - started).total_seconds()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_report(
            plan_a,
            plan_b,
            results_a,
            results_b,
            stats,
            population_total=population_total,
            elapsed=elapsed,
        )
    )

    return {
        "run_id": run_id,
        "population_total": population_total,
        "sample_a_n": len(plan_a.page_ids),
        "sample_b_n": len(plan_b.page_ids),
        "elapsed_seconds": round(elapsed, 1),
        "output_path": str(output_path),
        "raw_log_path": str(raw_log_path),
        "hallucination_rate": stats["hallucination_rate"],
    }


def _append_raw_log(path: Path, run_id: str, sample: str, analysis: PageAnalysis) -> None:
    """Append-only audit log of every LLM call (one JSON object per line)."""
    record = {
        "run_id": run_id,
        "sample": sample,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "page_id": analysis.page_id,
        "assigned_tags": analysis.assigned_tags,
        "rejected_assigned_tags": analysis.rejected_assigned_tags,
        "suggested_missing_categories": analysis.suggested_missing_categories,
        "confidence": analysis.confidence,
        "raw_response_sha256": hashlib.sha256(
            (analysis.raw_response or "").encode("utf-8")
        ).hexdigest(),
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Audit failures are non-fatal — the in-memory aggregation is the
        # source of truth for the report.
        pass
