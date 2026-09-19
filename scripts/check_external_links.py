#!/usr/bin/env python3
"""Check that the external links in the documentation still resolve.

Internal links are already covered by ``mkdocs build --strict``, which is fast and deterministic.
External links depend on other people's servers, so they are checked on a schedule instead: a
per-push check that fails because somebody else's site is briefly down teaches people to ignore
the gate.

    python scripts/check_external_links.py docs
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
TIMEOUT_SECONDS = 20
# Some servers refuse the default urllib agent outright.
HEADERS = {"User-Agent": "complexplorer-docs-link-check"}


def _collect(root: Path) -> dict[str, list[str]]:
    """Map each external URL to the pages that reference it."""
    found: dict[str, list[str]] = {}
    for page in sorted(root.rglob("*.md")):
        if "internal" in page.parts:  # not published, so not our problem
            continue
        for url in LINK.findall(page.read_text(encoding="utf-8")):
            found.setdefault(url.rstrip("."), []).append(str(page.relative_to(root)))
    return found


def _check(url: str) -> tuple[str, str | None]:
    """Return (url, failure) -- failure is None when the link is fine."""
    request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status < 400:
                return url, None
            failure = f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        # Plenty of servers reject HEAD but serve GET perfectly well.
        if error.code in (403, 405, 501):
            try:
                get = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(get, timeout=TIMEOUT_SECONDS) as response:
                    if response.status < 400:
                        return url, None
            except Exception as retry_error:  # noqa: BLE001 - reported, not handled
                return url, f"{error.code} on HEAD, then {type(retry_error).__name__} on GET"
        failure = f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001 - any failure is reportable
        failure = type(error).__name__
    return url, failure


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("docs")
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    links = _collect(root)
    print(f"  checking {len(links)} external link(s) under {root}/")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_check, links))

    failures = [(url, failure) for url, failure in results if failure]
    if failures:
        print(f"\n{len(failures)} external link(s) failed:", file=sys.stderr)
        for url, failure in sorted(failures):
            print(f"  - {url} ({failure})", file=sys.stderr)
            for page in links[url]:
                print(f"      referenced by {page}", file=sys.stderr)
        return 1

    print("\nall external links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
