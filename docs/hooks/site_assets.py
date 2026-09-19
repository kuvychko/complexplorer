"""Serve the example assets that live outside ``docs/``, without copying them into it.

The gallery images and the notebooks are produced by ``examples/showcase.py`` and live under
``examples/``. MkDocs only serves what is inside ``docs_dir``, and the generated gallery page
links ``../../examples/gallery/...`` in dozens of places.

Adding the files to the build here, rather than copying them into ``docs/`` or into ``site/``
afterwards, is what keeps ``--strict`` usable: MkDocs sees real files, so the references resolve
and a genuinely broken link still fails the build. Nothing is written into the working tree.

The destinations mirror the repository layout, which is also what the generated page expects: with
directory URLs the page is served at ``/gallery/gallery.generated/``, so ``../../examples/gallery/x``
resolves to ``/examples/gallery/x``. The page therefore needs no rewriting, and
``examples/showcase.py`` stays its only producer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.site_assets")

REPO_ROOT = Path(__file__).resolve().parents[2]

# (source directory relative to the repo, glob) pairs added to the site.
ASSET_SOURCES = [
    ("examples/gallery", "**/*.png"),
    ("examples/gallery", "**/*.jpg"),
    ("examples/gallery", "**/*.gif"),
    ("examples/gallery", "**/*.json"),
]

# The notebooks are deliberately NOT served. They carry their stored outputs, which is 26 MB of
# .ipynb, and nothing on the site links to them -- the pages that mention a notebook link to it on
# GitHub instead. Serving them also hands them to mkdocs-jupyter, whose conversion emits warnings
# that fail the strict build. Embedding them is a deliberate future choice, not an accident.


def on_files(files, config):
    """Add the example assets to the file set MkDocs is about to build."""
    added = 0
    seen: set[str] = set()

    for relative_dir, pattern in ASSET_SOURCES:
        source_dir = REPO_ROOT / relative_dir
        if not source_dir.is_dir():
            log.warning("site_assets: %s does not exist; skipping", relative_dir)
            continue

        for source in sorted(source_dir.glob(pattern)):
            # The path is given relative to the REPOSITORY root, so the file registers under
            # "examples/gallery/..." -- the same URI the pages link to. Registering it relative to
            # the asset directory instead would copy the file to the right place but leave link
            # validation unable to find it, which is a broken image with a passing build.
            relative_to_repo = source.relative_to(REPO_ROOT).as_posix()
            if relative_to_repo in seen:
                continue
            seen.add(relative_to_repo)
            files.append(
                File(
                    path=relative_to_repo,
                    src_dir=str(REPO_ROOT),
                    dest_dir=str(config["site_dir"]),
                    use_directory_urls=False,
                )
            )
            added += 1

    if not added:
        # A silent zero here would mean every gallery image on the site is broken.
        raise RuntimeError(
            "site_assets added no files; the gallery has not been generated "
            "(run `python examples/showcase.py`) or the paths have moved"
        )

    log.info("site_assets: added %d example asset(s)", added)
    return files
