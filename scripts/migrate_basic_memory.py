#!/usr/bin/env python3
"""Obsolete Basic Memory migration converter (direct writes disabled).

Converts frontmatter from Basic Memory format to Wiki format:
- Keeps title, maps updated from file mtime
- Strips permalink, tags, type fields
- Converts filename to kebab-case
- Preserves content and [[wiki-links]]

Use normal raw capture and ``chronovisor-sleep`` so a frontier-reviewed ingest
proposal owns any new knowledge page.
"""

import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.legacy_semantic_write import (  # noqa: E402
    block_legacy_semantic_mutation,
)


BASIC_MEMORY_ROOT = Path.home() / "basic-memory"
WIKI_PAGES = Path.home() / ".chronovisor" / "pages"
WIKI_RAW = Path.home() / ".chronovisor" / "raw"


def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    # Romanize Japanese characters won't work well, so use transliteration approach
    # For Japanese titles, use the permalink or generate from ascii parts
    # Remove non-ascii for slug, keep only alphanumeric and spaces
    ascii_text = ""
    for ch in text:
        if ch.isascii():
            ascii_text += ch
        else:
            ascii_text += " "

    # If mostly non-ascii, try to use a hash-based approach
    ascii_text = ascii_text.strip()
    if len(ascii_text) < 3:
        # Use permalink-style or hash
        import hashlib
        short_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        # Try to extract any meaningful ascii words
        words = [w for w in ascii_text.split() if w]
        if words:
            return "-".join(words).lower() + "-" + short_hash
        return short_hash

    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", ascii_text)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug.lower().strip("-")


def parse_basic_memory_frontmatter(content: str) -> tuple[dict, str]:
    """Parse Basic Memory frontmatter and return (metadata, body)."""
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    fm_text = content[3:end].strip()
    body = content[end + 3:].strip()

    fm = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()

    return fm, body


def convert_file(src: Path) -> tuple[str, str] | None:
    """Convert a Basic Memory file to Wiki format.

    Returns (filename, content) or None if should be skipped.
    """
    try:
        raw_content = src.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    if not raw_content.strip():
        return None

    fm, body = parse_basic_memory_frontmatter(raw_content)

    # Get title
    title = fm.get("title", src.stem)

    # Generate filename from permalink or title
    permalink = fm.get("permalink", "")
    if permalink:
        # Use last part of permalink path
        slug = permalink.rstrip("/").split("/")[-1]
        slug = re.sub(r"[^a-zA-Z0-9-]", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
    else:
        slug = slugify(title)

    if not slug:
        slug = slugify(src.stem)

    if not slug:
        return None

    filename = f"{slug}.md"

    # Get updated date from file mtime
    mtime = datetime.fromtimestamp(src.stat().st_mtime)
    updated = mtime.strftime("%Y-%m-%d")

    # Build new content
    new_content = f"---\ntitle: {title}\nupdated: {updated}\n---\n\n{body}\n"

    return filename, new_content


def migrate():
    """Run the migration."""
    block_legacy_semantic_mutation(
        tool="migrate_basic_memory.py",
        replacement="the normal raw capture plus chronovisor-sleep ingest lane",
    )
    WIKI_PAGES.mkdir(parents=True, exist_ok=True)

    all_files = list(BASIC_MEMORY_ROOT.rglob("*.md"))
    print(f"Found {len(all_files)} Basic Memory files")

    migrated = 0
    skipped = 0
    conflicts = 0
    seen_filenames = {}

    for src in all_files:
        result = convert_file(src)
        if result is None:
            skipped += 1
            continue

        filename, content = result

        # Handle filename conflicts
        if filename in seen_filenames:
            # Append a counter
            base = filename.rsplit(".", 1)[0]
            counter = 2
            while f"{base}-{counter}.md" in seen_filenames:
                counter += 1
            filename = f"{base}-{counter}.md"
            conflicts += 1

        seen_filenames[filename] = src

        dest = WIKI_PAGES / filename

        # Don't overwrite existing wiki pages
        if dest.exists():
            skipped += 1
            continue

        dest.write_text(content, encoding="utf-8")
        migrated += 1

    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print(f"Conflicts resolved: {conflicts}")
    print(f"Total pages in wiki: {len(list(WIKI_PAGES.glob('*.md')))}")


if __name__ == "__main__":
    migrate()
