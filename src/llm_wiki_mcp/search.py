"""Search engine — BM25 + semantic search with RRF fusion."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from llm_wiki_mcp.wiki import WIKI_ROOT, PAGES_DIR, all_pages, page_id_from_path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ScoredPage:
    page_id: str
    title: str
    folder: str
    updated: str
    score: float
    snippet: str = ""


# ---------------------------------------------------------------------------
# Tokenizer (no external dependency)
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uff66", "\uff9f"),  # Halfwidth Katakana
)

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _is_cjk(ch: str) -> bool:
    for lo, hi in _CJK_RANGES:
        if lo <= ch <= hi:
            return True
    return False


def tokenize(text: str) -> list[str]:
    """Tokenize text: ASCII words + CJK character bigrams (boundary-aware)."""
    # Strip frontmatter
    text = _FRONTMATTER_RE.sub("", text)
    text_lower = text.lower()

    tokens = []
    # ASCII words
    for m in re.finditer(r"[a-z0-9_]+", text_lower):
        word = m.group()
        if len(word) >= 2:
            tokens.append(word)

    # CJK bigrams — boundary-aware (don't cross non-CJK gaps)
    cjk_runs = re.findall(r"[" + "".join(f"{lo}-{hi}" for lo, hi in _CJK_RANGES) + r"]+", text)
    for run in cjk_runs:
        # Single CJK character: include as unigram
        if len(run) == 1:
            tokens.append(run)
        # Bigrams within each contiguous CJK run
        for i in range(len(run) - 1):
            tokens.append(run[i] + run[i + 1])

    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index built on-the-fly."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[str, str, str, str, list[str]]] = []  # (page_id, title, folder, updated, tokens)
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n: int = 0

    def build(self) -> None:
        """Build index from all wiki pages."""
        self._docs = []
        self._df = {}
        total_len = 0

        for path in all_pages():
            content = path.read_text()
            fm_match = re.search(r"title:\s*(.+)", content)
            title = fm_match.group(1).strip() if fm_match else path.stem
            updated_match = re.search(r"updated:\s*(.+)", content)
            updated = updated_match.group(1).strip() if updated_match else ""
            folder = path.parent.name if path.parent != PAGES_DIR else ""

            # Tokenize with title boost (repeat title tokens)
            title_tokens = tokenize(title) * 3
            body_tokens = tokenize(content)
            all_tokens = title_tokens + body_tokens

            self._docs.append((page_id_from_path(path), title, folder, updated, all_tokens))
            total_len += len(all_tokens)

            seen = set(all_tokens)
            for tok in seen:
                self._df[tok] = self._df.get(tok, 0) + 1

        self._n = len(self._docs)
        self._avgdl = total_len / self._n if self._n else 1.0

    def query(self, query_text: str, top_n: int = 20) -> list[ScoredPage]:
        """Search the index."""
        if not self._docs:
            self.build()

        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []

        results = []
        for page_id, title, folder, updated, doc_tokens in self._docs:
            score = 0.0
            dl = len(doc_tokens)

            # Count term frequencies
            tf_map: dict[str, int] = {}
            for tok in doc_tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                df = self._df.get(qt, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                score += idf * tf_norm

            if score > 0:
                results.append(ScoredPage(
                    page_id=page_id, title=title, folder=folder,
                    updated=updated, score=score,
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]


# ---------------------------------------------------------------------------
# Semantic search (Ollama embeddings)
# ---------------------------------------------------------------------------

EMBEDDINGS_FILE = WIKI_ROOT / ".embeddings.json"
EMBED_MODEL = "nomic-embed-text"


def _load_embeddings() -> dict:
    if EMBEDDINGS_FILE.exists():
        return json.loads(EMBEDDINGS_FILE.read_text())
    return {}


def _save_embeddings(data: dict) -> None:
    EMBEDDINGS_FILE.write_text(json.dumps(data, ensure_ascii=False))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def update_embeddings(page_ids: list[str] | None = None) -> int:
    """Update embeddings for pages. Returns count of updated pages."""
    from llm_wiki_mcp.ollama import embed, is_available

    if not is_available():
        return 0

    store = _load_embeddings()
    updated_count = 0

    pages_to_process = []
    for path in all_pages():
        pid = page_id_from_path(path)
        if page_ids and pid not in page_ids:
            continue

        content = path.read_text()
        mtime = path.stat().st_mtime

        existing = store.get(pid)
        if existing and existing.get("mtime", 0) >= mtime and not page_ids:
            continue

        # Truncate content for embedding (first ~2000 chars)
        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1).strip() if fm_match else pid
        embed_text = f"{title}\n\n{_FRONTMATTER_RE.sub('', content)[:2000]}"
        pages_to_process.append((pid, embed_text, mtime))

    # Batch embed
    if pages_to_process:
        batch_size = 32
        for i in range(0, len(pages_to_process), batch_size):
            batch = pages_to_process[i:i + batch_size]
            texts = [t[1] for t in batch]
            try:
                vectors = embed(texts)
                for (pid, _, mtime), vec in zip(batch, vectors):
                    store[pid] = {"vector": vec, "mtime": mtime}
                    updated_count += 1
            except Exception:
                continue

    _save_embeddings(store)
    return updated_count


def semantic_search(query: str, top_n: int = 20) -> list[ScoredPage]:
    """Search using embedding similarity."""
    from llm_wiki_mcp.ollama import embed, is_available

    if not is_available():
        return []

    store = _load_embeddings()
    if not store:
        return []

    try:
        q_vec = embed([query])[0]
    except Exception:
        return []

    # Build page metadata map
    page_meta = {}
    for path in all_pages():
        pid = page_id_from_path(path)
        content = path.read_text()
        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1).strip() if fm_match else pid
        updated_match = re.search(r"updated:\s*(.+)", content)
        updated = updated_match.group(1).strip() if updated_match else ""
        folder = path.parent.name if path.parent != PAGES_DIR else ""
        page_meta[pid] = (title, folder, updated)

    results = []
    for pid, data in store.items():
        if pid not in page_meta:
            continue
        vec = data.get("vector")
        if not vec:
            continue
        sim = _cosine_sim(q_vec, vec)
        title, folder, updated = page_meta[pid]
        results.append(ScoredPage(
            page_id=pid, title=title, folder=folder,
            updated=updated, score=sim,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# RRF Fusion + Filter + Sort
# ---------------------------------------------------------------------------

def fuse_results(
    bm25_results: list[ScoredPage],
    semantic_results: list[ScoredPage],
    k: int = 60,
) -> list[ScoredPage]:
    """Reciprocal Rank Fusion of two result lists."""
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}

    for rank, page in enumerate(bm25_results):
        scores[page.page_id] = scores.get(page.page_id, 0) + 1 / (k + rank)
        meta[page.page_id] = page

    for rank, page in enumerate(semantic_results):
        scores[page.page_id] = scores.get(page.page_id, 0) + 1 / (k + rank)
        if page.page_id not in meta:
            meta[page.page_id] = page

    fused = []
    for pid, score in scores.items():
        p = meta[pid]
        fused.append(ScoredPage(
            page_id=p.page_id, title=p.title, folder=p.folder,
            updated=p.updated, score=score,
        ))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def apply_filters(
    results: list[ScoredPage],
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> list[ScoredPage]:
    """Filter results by folder and date range."""
    filtered = results
    if folder:
        filtered = [r for r in filtered if r.folder.startswith(folder)]
    if updated_after:
        filtered = [r for r in filtered if r.updated >= updated_after and r.updated != "unknown"]
    if updated_before:
        filtered = [r for r in filtered if r.updated <= updated_before and r.updated != "unknown"]
    return filtered


def apply_sort(results: list[ScoredPage], sort_by: str = "relevance") -> list[ScoredPage]:
    """Sort results."""
    if sort_by == "date":
        return sorted(results, key=lambda x: x.updated, reverse=True)
    elif sort_by == "title":
        return sorted(results, key=lambda x: x.title)
    return results  # relevance is already sorted


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
) -> tuple[list[ScoredPage], str]:
    """Run search and return (results, search_mode)."""
    # Fetch more results before filtering to avoid truncation-before-filter bug
    fetch_n = max(top_n * 5, 100)

    bm25 = BM25Index()
    bm25.build()
    bm25_results = bm25.query(query, top_n=fetch_n)

    search_mode = "bm25"
    if semantic:
        sem_results = semantic_search(query, top_n=fetch_n)
        if sem_results:
            results = fuse_results(bm25_results, sem_results)
            search_mode = "hybrid"
        else:
            results = bm25_results
    else:
        results = bm25_results

    # Filter THEN truncate (not the other way around)
    results = apply_filters(results, folder, updated_after, updated_before)
    results = apply_sort(results, sort_by)
    return results[:top_n], search_mode
