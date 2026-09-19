"""Tell the reader which version of complexplorer the page in front of them documents.

The site replaces the published 2.x documentation at one URL, so a reader has no other way to
tell a released page from one built off an untagged commit. The version comes from the installed
package rather than from the repository, because the installed package is what the examples on
these pages were checked against.
"""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger("mkdocs.hooks.version_banner")


def _release_tag() -> str | None:
    """The exact ``v*`` tag at HEAD, or None when this is not a tagged build."""
    try:
        finished = subprocess.run(
            ["git", "describe", "--exact-match", "--tags", "--match", "v*"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # no git, or no repository
        return None
    return finished.stdout.strip() or None if finished.returncode == 0 else None


def on_config(config):
    """Put the version and its release state where the theme override can read them."""
    try:
        from complexplorer import __version__
    except ImportError as error:  # the docs environment does not match the code
        raise RuntimeError(
            "version_banner cannot import complexplorer; install the package "
            "into the environment building the docs"
        ) from error

    tag = _release_tag()
    config["extra"]["complexplorer_version"] = __version__
    config["extra"]["is_release_build"] = tag is not None

    log.info("version_banner: %s (%s)", __version__, tag or "development build")
    return config
