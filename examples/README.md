# Complexplorer Examples

Tutorials, runnable demos, and the rendered gallery for the complexplorer library.

## 📁 Layout

```
examples/
├── notebooks/   Jupyter tutorials (start here)
├── scripts/     runnable Python demos (best 3D quality — run from a terminal)
├── gallery/     rendered gallery images (regenerated from the preset registry)
└── README.md    this file
```

> **Backend policy (3.0):** matplotlib powers **2D**; **PyVista powers all 3D** (landscapes,
> Riemann sphere/relief/surface, STL export). The legacy matplotlib 3D functions
> (`plot_landscape`, `pair_plot_landscape`, the 3D `riemann`) were removed in 3.0 — use the
> `*_pv` functions (`plot_landscape_pv`, `pair_plot_landscape_pv`, `riemann_pv`,
> `riemann_surface_pv`). PyVista is a required dependency.

## 📚 Notebooks (`notebooks/`)

| Notebook | What it covers |
|---|---|
| `getting_started.ipynb` | Installation, your first visualization, domains and colormaps, basic 2D/3D — **start here.** |
| `advanced_features.ipynb` | Phase portraits, the colormap family, PyVista 3D, Riemann sphere **and Riemann surfaces**. |
| `stl_export_demo.ipynb` | Step-by-step 3D-printable ornaments: Riemann-sphere relief, scaling options, print tips. |
| `api_cookbook.ipynb` | Common functions, domain/colormap patterns, the **preset registry** (`cp.catalog`), recipes. |

> **PyVista in Jupyter:** each notebook sets the **static** backend
> (`pv.set_jupyter_backend('static')`) so 3D plots embed as images and the notebook runs
> headlessly. For interactive rotation/zoom and the best anti-aliasing, run the terminal
> scripts below instead.

### Running & verifying the notebooks

Install the notebook tooling, then verify all four execute top-to-bottom:

```bash
uv pip install -e ".[examples]"          # nbmake, nbconvert, ipykernel
pytest --nbmake examples/notebooks/       # the local execution gate (opt-in; not in the default suite or CI)
```

To regenerate the committed output (e.g. after editing a notebook):

```bash
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb
```

## 🖥️ Scripts (`scripts/`)

### `interactive_showcase.py` — menu-driven explorer

A command-line interface to explore complex functions with high-quality PyVista 3D output
(2D phase portraits, 3D landscapes, Riemann sphere). Run it from a terminal for the best
rendering quality:

```bash
python examples/scripts/interactive_showcase.py
```

**Interactive window controls:** left-drag rotate · middle-drag pan · right-drag / scroll
zoom · `R` reset camera · `S` screenshot · `Q` close. Start at a lower resolution to explore
quickly, then raise it for a final render. All 3D views include Re/Im/Z orientation axes.

> **Tip:** PyVista renders best from a terminal, **not** inside Jupyter — command-line scripts
> get superior antialiasing and full interactivity.

## 🎨 Gallery (`gallery/`)

The gallery is **generated from the curated preset registry** (`cp.catalog`) rather than
hand-maintained. The library is the single source of truth — there is no separate hand-rolled
generator script here. Render a bundle with the CLI:

```bash
complexplorer gallery --tag <tag> -o gallery_output    # or: -i <id> ...
```

This writes a deterministic `index.json` manifest plus per-preset portraits and `card.json`
records. (Higher-resolution 3D / Riemann / STL gallery renders are produced by the
`examples/showcase.py` script — added in a follow-up change.)

## 🖨️ STL export (3D printing)

```python
from complexplorer.export.stl import OrnamentGenerator

generator = OrnamentGenerator(lambda z: z / (z**10 - 1), resolution=150)
generator.generate_and_save("ornament.stl", size_mm=80)
```

See `notebooks/stl_export_demo.ipynb` for the full guide.

## 🐛 Troubleshooting

- **`No module named complexplorer`** — install it: `pip install -e .` (from the repo root).
- **PyVista window doesn't appear** — pass `notebook=False` in Jupyter; ensure a display/GPU
  with OpenGL is available; try `pip install -U pyvista`.
- **Low-quality 3D in Jupyter** — expected; the inline backend aliases badly. Use
  `notebook=False` for an external window, or run the terminal scripts.

## 🔗 Resources

- [Complexplorer on GitHub](https://github.com/kuvychko/complexplorer)
- [Visual Complex Functions](https://link.springer.com/book/10.1007/978-3-0348-0180-5) — Elias Wegert
- [PyVista documentation](https://docs.pyvista.org/)
