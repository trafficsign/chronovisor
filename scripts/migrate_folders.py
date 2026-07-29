#!/usr/bin/env python3
"""Diagnose legacy folder assignments; semantic moves are disabled."""

import re
import json
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.core.store import PAGES_DIR, all_pages
from chronovisor.core.ollama import generate
from chronovisor.raw.legacy_semantic_write import block_legacy_semantic_mutation

BATCH_SIZE = 100

CLASSIFY_PROMPT = """\
You are a file organizer. Classify each wiki page into exactly one folder category.

Rules:
- Use short, broad English kebab-case folder names
- Keep folder depth to 1 level only
- Respond with JSON array: [{"page_id": "...", "folder": "..."}]
- Do NOT include the .md extension in page_id
- Do NOT create overly specific folders. Keep categories broad.

Suggested folder categories (create new ones if needed):
- career/ — job change, career planning, interviews, work culture
- project/ — software projects (jttk, sonar, vis, dwgdb, etc.)
- auto-industry/ — automotive industry, Mazda, EV, manufacturing
- ai/ — AI tools, LLM, Claude, Codex, memory systems
- hardware/ — computers, displays, peripherals
- cad/ — CAD/CAE tools, formats (JT, Parasolid, NX)
- engineering/ — mechanical engineering, structural analysis, materials
- spec/ — design specifications, part numbers, requirements
- misc/ — anything that doesn't fit above

Pages to classify:
"""


def get_page_titles() -> list[dict]:
    """Get all page IDs and titles."""
    pages = []
    for path in sorted(all_pages()):
        content = path.read_text()
        title_match = re.search(r"title:\s*(.+)", content)
        title = title_match.group(1).strip() if title_match else path.stem
        pages.append({"page_id": path.stem, "title": title, "path": path})
    return pages


def classify_batch(batch: list[dict]) -> list[dict]:
    """Send a batch to Gemma4 for classification."""
    lines = [f"- {p['page_id']}: {p['title']}" for p in batch]
    prompt = CLASSIFY_PROMPT + "\n".join(lines)

    output = generate(prompt, system=None)

    # Extract JSON from output
    json_match = re.search(r"\[.*\]", output, re.DOTALL)
    if not json_match:
        print(f"  WARNING: Could not parse JSON from output")
        print(f"  Raw output: {output[:500]}")
        return []

    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}")
        return []


def move_page(path: Path, folder: str) -> bool:
    """Move a page to a subfolder."""
    block_legacy_semantic_mutation(
        tool="migrate_folders.py",
        replacement="chronovisor-sleep",
    )
    target_dir = PAGES_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name

    if target.exists():
        print(f"  SKIP: {target} already exists")
        return False

    shutil.move(str(path), str(target))
    return True


def main():
    if "--apply" in sys.argv:
        block_legacy_semantic_mutation(
            tool="migrate_folders.py",
            replacement="chronovisor-sleep",
        )
    pages = get_page_titles()
    print(f"Total pages: {len(pages)}")

    # Skip pages already in subfolders
    flat_pages = [p for p in pages if p["path"].parent == PAGES_DIR]
    print(f"Flat pages to migrate: {len(flat_pages)}")

    if not flat_pages:
        print("No flat pages to migrate.")
        return

    # Process in batches
    all_assignments = []
    for i in range(0, len(flat_pages), BATCH_SIZE):
        batch = flat_pages[i:i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}/{(len(flat_pages) + BATCH_SIZE - 1) // BATCH_SIZE}: {len(batch)} pages")

        assignments = classify_batch(batch)
        print(f"  Classified: {len(assignments)} pages")

        all_assignments.extend(assignments)

    # Summary
    folders = {}
    for a in all_assignments:
        folder = a.get("folder", "misc")
        folders[folder] = folders.get(folder, 0) + 1

    print(f"\n=== Classification Summary ===")
    for folder, count in sorted(folders.items(), key=lambda x: -x[1]):
        print(f"  {folder}/: {count} pages")
    print(f"  Total: {sum(folders.values())} pages")

    # Save assignments for review before moving
    assignments_file = Path(__file__).parent / "folder_assignments.json"
    with open(assignments_file, "w") as f:
        json.dump(all_assignments, f, indent=2, ensure_ascii=False)
    print(f"\nAssignments saved to {assignments_file}")
    print("Diagnostic artifact only; chronovisor-sleep owns any semantic move decision.")

    if "--apply" in sys.argv:
        print("\n=== Applying moves ===")
        # Build lookup
        path_lookup = {p["page_id"]: p["path"] for p in flat_pages}
        moved = 0
        for a in all_assignments:
            page_id = a.get("page_id", "")
            folder = a.get("folder", "misc")
            path = path_lookup.get(page_id)
            if path and path.exists():
                if move_page(path, folder):
                    moved += 1
        print(f"Moved {moved} pages.")


if __name__ == "__main__":
    main()
