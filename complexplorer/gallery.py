"""Gallery generator — render catalog presets into a reproducible asset bundle.

For each selected preset this writes a 2D portrait PNG and a ``card.json``, plus a
self-contained top-level ``index.json`` manifest. The **manifest** is the deterministic
contract (byte-identical across runs of the same library version) and the interchange record
consumed by downstream tools (Godot game prototyping, a future docs/web layer, the Phase 3
level-export). Portrait images are reproducible best-effort (metadata stripped).

matplotlib-only, PyVista-free.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

from ._version import __version__
from .core.presets import FunctionPreset, catalog
from .exceptions import ValidationError
from .plotting.matplotlib.plot_2d import plot as plot_2d

SCHEMA_VERSION = 2  # v2: card records carry answer_key_stats (enrich-answer-key-stats)
_FIGSIZE = (4.0, 4.0)


def _resolve(selection: str | Iterable[str] | None) -> list[FunctionPreset]:
    """Resolve a selection (a tag, an iterable of ids, or None=all) to id-sorted presets."""
    if selection is None:
        presets = [catalog.get(i) for i in catalog.list()]
    elif isinstance(selection, str):  # a tag
        presets = catalog.filter(tag=selection)
    else:  # an iterable of ids
        presets = [catalog.get(i) for i in selection]
    return sorted(presets, key=lambda p: p.id)


def _write_json(path: Path, obj: dict) -> None:
    """Write JSON deterministically: sorted keys, stable indent, trailing newline, LF."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def _render_portrait(preset: FunctionPreset, path: Path, dpi: int) -> None:
    """Render the preset's 2D phase portrait to ``path`` with metadata stripped.

    Owns the figure (``plt.subplots`` + an explicit ``ax``) so renders never accumulate on
    the global pyplot axes.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    try:
        plot_2d(preset.domain(), preset.func, cmap=preset.colormap(), ax=ax)
        fig.savefig(path, dpi=dpi, metadata={"Software": None})
    finally:
        plt.close(fig)


def _card(preset: FunctionPreset) -> dict:
    """The per-preset record: ``to_dict()`` + schema version + a relative file map."""
    rec = preset.to_dict()
    rec["schema_version"] = SCHEMA_VERSION
    # Forward-slash relative path: OS-independent, so the manifest is identical on any platform.
    rec["files"] = {"portrait": f"{preset.id}/portrait.png"}
    return rec


def generate_gallery(
    out_dir: str | Path, *, selection: str | Iterable[str] | None = None, dpi: int = 150
) -> dict:
    """Render a selection of catalog presets into ``out_dir`` and return the manifest.

    Parameters
    ----------
    out_dir : path
        Directory to write the bundle into (created if needed).
    selection : str | iterable of str | None
        A tag, an iterable of preset ids, or None for the whole catalog.
    dpi : int
        Portrait resolution.

    Returns
    -------
    dict
        The ``index.json`` manifest that was written.
    """
    presets = _resolve(selection)
    if not presets:
        raise ValidationError(f"0 presets matched selection {selection!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for p in presets:
        pdir = out / p.id
        pdir.mkdir(exist_ok=True)
        _render_portrait(p, pdir / "portrait.png", dpi)
        rec = _card(p)
        _write_json(pdir / "card.json", rec)
        records.append(rec)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complexplorer_version": __version__,
        "generator": "complexplorer gallery",
        "presets": records,  # already id-sorted
    }
    _write_json(out / "index.json", manifest)
    return manifest
