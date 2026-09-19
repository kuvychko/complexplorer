# Complexplorer: Phased Implementation Plan

**Project:** Complexplorer 2.x → 3.x roadmap  
**Repository:** <https://github.com/kuvychko/complexplorer>  
**Prepared for:** Igor Kuvychko  
**Date:** 2026-06-20  
**Format:** Implementation roadmap for Obsidian / GitHub planning

---

## 1. Executive Summary

Complexplorer should evolve from a domain-coloring package into a **creative computing toolkit for complex functions, complex geometry, engineering transfer functions, and physical mathematical artifacts**.

The current library already has a distinctive center of gravity:

- enhanced phase portraits,
- Riemann relief maps,
- PyVista-powered 3D rendering,
- STL export,
- advanced colormaps,
- domain composition,
- and a promising “math to matter” story.

The next step is not merely “more plotting functions.” The next step is to make Complexplorer feel like a **laboratory**:

> Define a complex object, explore it visually, interact with it, learn from it, export it, and potentially turn it into a physical object or website project.

The most promising expansion directions are:

1. **PyVista-first 3D visualization architecture**  
   Advanced 3D should become PyVista-only. Matplotlib remains excellent for 2D plots, but Riemann surfaces, relief maps, multi-sheet geometry, and printable meshes should not be forced through matplotlib’s 3D stack.

2. **Riemann surface and branch-cut visualization**  
   Move beyond single-plane domain coloring into multi-sheeted surfaces, branch points, monodromy, and algebraic curves.

3. **Interactive games and educational demos**  
   Create playful tools such as Singularity Detective, Branch-Cut Zoo, Function Guessr, Monodromy Maze, and Möbius Playground.

4. **Electrical engineering / resonator / filter design applications**  
   Use complex-domain visualization to explore transfer functions, poles, zeros, Bode/Nyquist plots, RF networks, impedance spectra, QCM resonators, and filter response surfaces.

5. **Objects / Projects integration**  
   Turn selected visualizations into durable personal-site objects: printable artifacts, visual essays, interactive demos, Printables-linked sculptures, and project cards.

The strongest product thesis:

> **Complexplorer makes complex-valued structure visible, explorable, and physical.**

---

## 2. Current-State Assumptions

This plan assumes the current public repo state around v2.0:

- package name: `complexplorer`
- Python 3.11+
- core dependencies: NumPy, matplotlib, SciPy
- optional extras: `pyvista`, `qt`, `all`
- current public positioning: complex function visualization, Riemann relief maps, STL export, PyVista integration, advanced colormaps, domain composition, and documentation/gallery support

Current packaging note:

- README/license presentation appears MIT-oriented.
- `pyproject.toml` currently classifies the project as BSD licensed.
- PyVista is currently optional via `complexplorer[pyvista]`.

This plan recommends that **PyVista become a required dependency no later than the 3.0 line**, because the next serious growth path is fundamentally 3D and mesh-oriented.

---

## 3. Product North Star

### 3.1 One-sentence positioning

**Complexplorer is a Python toolkit for visualizing, exploring, and fabricating complex-valued mathematical and engineering structures.**

### 3.2 Three audience modes

| Mode | Audience | What they want |
|---|---|---|
| **Research / math** | Mathematicians, students, scientific programmers | phase portraits, Riemann surfaces, branch structure, special functions |
| **Engineering** | EE, RF, controls, resonator, signal-processing users | poles, zeros, transfer functions, impedance, filters, stability, frequency response |
| **Creative / public** | personal website visitors, educators, makers | interactive demos, games, printable objects, visual essays |

### 3.3 Product identity

Complexplorer should not compete with general plotting libraries. It should specialize in the visual grammar of complex functions:

- phase,
- modulus,
- zeros,
- poles,
- branch points,
- cuts,
- sheets,
- maps,
- surfaces,
- and physical relief.

---

## 4. Key Architecture Decision: PyVista-First 3D

### 4.1 Recommendation

Make PyVista the required 3D backend and stop trying to maintain feature parity between matplotlib 3D and PyVista.

Proposed policy:

| Visualization type | Backend |
|---|---|
| 2D phase portraits | matplotlib |
| 2D pair plots | matplotlib |
| 2D static educational figures | matplotlib |
| 3D analytic landscapes | PyVista |
| Riemann relief maps | PyVista |
| Riemann surfaces | PyVista |
| branch-cut sheet geometry | PyVista |
| STL / mesh export | PyVista-oriented mesh pipeline |
| high-quality screenshots / movies | PyVista |

### 4.2 Why this matters

Advanced 3D is not just “plotting with a z-axis.” It becomes a mesh, camera, lighting, clipping, scalar-field, texture, and export problem.

Trying to support advanced 3D equally in matplotlib and PyVista would create several costs:

- duplicated APIs,
- weaker 3D interactions,
- lower rendering quality,
- more edge cases,
- weaker mesh export path,
- and pressure to simplify the mathematical design to fit the weaker backend.

The better principle:

> Use matplotlib where matplotlib is excellent; use PyVista where the problem is actually a 3D mesh/geometry problem.

### 4.3 Dependency migration strategy

Do not surprise existing users in a patch release. Use a staged transition.

#### Version 2.1

- Keep PyVista optional.
- Add warnings in advanced 3D docs: “PyVista is the primary 3D backend.”
- Mark matplotlib 3D as legacy / basic / static.
- Add tests that verify PyVista workflows when installed.

#### Version 2.2

- Add `complexplorer[3d]` alias to `complexplorer[pyvista]`.
- Start moving new 3D functionality exclusively into PyVista modules.
- Add explicit documentation: “New 3D features are PyVista-only.”

#### Version 3.0

- Make PyVista a required dependency.
- Remove or freeze matplotlib 3D advanced paths.
- Keep matplotlib as the 2D backend.
- Simplify user-facing docs: no more “optional 3D backend” ambiguity.

### 4.4 Proposed package extras after transition

```toml
[project]
dependencies = [
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "scipy>=1.11.0",
    "pyvista>=0.45.0",
]

[project.optional-dependencies]
qt = ["PyQt6>=6.5.0"]
ee = ["scikit-rf>=1.0", "control>=0.10"]
dev = ["pytest", "pytest-cov", "pytest-timeout", "ruff", "mypy"]
docs = ["mkdocs", "mkdocs-material", "mkdocstrings[python]"]
all = ["complexplorer[qt,ee,dev,docs]"]
```

Potential alternative:

- Keep a future `complexplorer-lite` for 2D-only users.
- But do not complicate the main package unless user demand appears.

---

## 5. Proposed High-Level Architecture

### 5.1 Module layout

```text
complexplorer/
  core/
    domains.py
    grids.py
    fields.py
    sampling.py
    scaling.py
    singularities.py
    presets.py

  colormaps/
    phase.py
    perceptual.py
    artistic.py
    accessibility.py

  plot2d/
    phase_portrait.py
    pair_plot.py
    annotations.py

  plot3d/
    pyvista_backend.py
    landscapes.py
    riemann_sphere.py
    relief.py
    cameras.py
    lighting.py
    screenshots.py

  surfaces/
    sheet.py
    branch_points.py
    branch_cuts.py
    riemann_surface.py
    algebraic_curve.py
    monodromy.py
    examples.py

  ee/
    transfer_function.py
    poles_zeros.py
    bode.py
    nyquist.py
    impedance.py
    touchstone.py
    resonators.py
    qcm.py

  games/
    levels.py
    singularity_detective.py
    branch_cut_zoo.py
    function_guessr.py
    monodromy_maze.py

  export/
    stl.py
    obj.py
    gltf.py
    png.py
    html.py
    project_card.py

  cli/
    main.py
    render.py
    gallery.py
    stl.py
    game_assets.py

  web/
    asset_manifest.py
    static_export.py
```

### 5.2 Core conceptual objects

#### `ComplexDomain`

Existing domain classes should remain central.

Examples:

```python
cp.Rectangle(width=4, height=4)
cp.Disk(radius=2)
cp.Annulus(inner_radius=0.2, outer_radius=3)
cp.Domain.union(...)
cp.Domain.difference(...)
```

#### `ComplexGrid`

A sampled grid in the complex plane.

```python
grid = cp.ComplexGrid.from_domain(domain, resolution=800)
```

Responsibilities:

- hold `Z`, mask, bounds, resolution,
- cache coordinate transforms,
- represent domain restriction cleanly,
- support decimation for fast preview.

#### `ComplexField`

A sampled complex-valued function over a domain.

```python
field = cp.ComplexField.from_function(f, grid)
```

Responsibilities:

- store values `W = f(Z)`,
- handle infinities and NaNs,
- compute phase/modulus,
- detect zeros/poles heuristically,
- expose metadata for plotting and games.

#### `SurfaceMesh`

A PyVista-oriented mesh with attached scalar/color fields.

```python
mesh = cp.SurfaceMesh.from_riemann_relief(field, mode="arctan")
```

Responsibilities:

- hold vertices, faces, scalar fields,
- validate mesh quality,
- export STL/OBJ/glTF,
- support clipping, smoothing, decimation,
- support printability checks.

#### `FieldOnSurface`

A complex field defined on a geometric surface.

```python
surface = cp.RiemannSurface.sqrt(domain)
field = surface.evaluate(lambda z, w: w)
```

This becomes the backbone of Riemann-surface visualization.

---

## 6. Phased Roadmap

## Phase 0 — Repo Hygiene and Foundation

**Goal:** Make the current package easier to extend before adding new conceptual machinery.

**Recommended timeframe:** 1–2 focused weeks  
**Release target:** v2.1.0

### 0.1 Deliverables

- Fix package metadata and license mismatch.
- Normalize raw markdown and source formatting.
- Add `ruff` and possibly `mypy` or `pyright` for lightweight static checks.
- Add an explicit backend policy document.
- Add architecture notes for 2D vs 3D responsibilities.
- Add a “roadmap” page to docs.
- Create top-level issue labels:
  - `2d`
  - `3d-pyvista`
  - `riemann-surfaces`
  - `games`
  - `ee`
  - `objects-projects`
  - `docs`
  - `api-design`

### 0.2 Specific tasks

#### Packaging

- Align README, `LICENSE`, and `pyproject.toml` classifier.
- Decide whether the public license is MIT or BSD; update all metadata accordingly.
- Add `project.scripts` for CLI entry point:

```toml
[project.scripts]
complexplorer = "complexplorer.cli.main:main"
```

#### Formatting

- Add `ruff format`.
- Add GitHub Actions job for:
  - lint,
  - tests,
  - doc build,
  - optional PyVista test matrix.

#### Dependency policy

- For v2.1, keep PyVista optional.
- Add documentation stating that new advanced 3D features will be PyVista-only.
- Add a deprecation warning for advanced matplotlib 3D paths if they exist.

### 0.3 Acceptance criteria

- `pip install complexplorer` works.
- `pip install "complexplorer[pyvista]"` works.
- Docs clearly say what is 2D matplotlib and what is 3D PyVista.
- CI tests both “base” and “with PyVista” configurations.
- The README no longer overclaims or has contradictory license metadata.

### 0.4 Why this phase matters

Riemann surfaces, games, and EE integrations will all add API surface area. If the existing package metadata, docs, and backend boundaries are fuzzy, every future feature will amplify that fuzziness.

---

## Phase 1 — PyVista 3D Kernel

**Goal:** Create a clean internal 3D foundation that can support relief maps, Riemann surfaces, mesh export, and interactive objects.

**Recommended timeframe:** 2–4 weeks  
**Release target:** v2.2.0

### 1.1 Deliverables

- `complexplorer.plot3d` package.
- Unified PyVista mesh-generation pipeline.
- Camera presets.
- Lighting presets.
- Screenshot/export helpers.
- Mesh metadata model.
- Documentation page: “3D Visualization Architecture.”

### 1.2 API sketch

```python
import complexplorer as cp

f = lambda z: z / (z**10 - 1)
domain = cp.Annulus(0.2, 3.0)

field = cp.sample(f, domain, resolution=600)
mesh = cp.mesh.riemann_relief(field, modulus_mode="arctan")

cp.show(mesh, camera="isometric", lighting="studio")
cp.save_screenshot(mesh, "riemann_relief.png", size=(2400, 1600))
cp.export_stl(mesh, "riemann_relief.stl")
```

### 1.3 Required design decisions

#### Mesh ownership

Avoid hiding all PyVista objects from advanced users. Instead:

- expose high-level Complexplorer wrappers,
- but allow access to the underlying `pyvista.PolyData`.

Example:

```python
mesh = cp.mesh.riemann_relief(field)
pv_mesh = mesh.to_pyvista()
```

#### Color ownership

Attach colors/scalars to mesh objects, not just plot calls.

```python
mesh.attach_phase_colors(cmap=cp.Phase())
mesh.attach_scalar("log_modulus", np.log1p(np.abs(field.values)))
```

#### Camera presets

Add named camera settings:

- `topographic`
- `isometric`
- `riemann_sphere_front`
- `riemann_sphere_cutaway`
- `gallery_thumbnail`
- `print_preview`

#### Lighting presets

Add named lighting settings:

- `studio`
- `matte`
- `topographic`
- `dramatic`
- `printable_object`

### 1.4 Implementation tasks

- Refactor existing `riemann_pv`, `plot_landscape_pv`, and `pair_plot_landscape_pv` into shared mesh utilities.
- Add `SurfaceMesh` wrapper.
- Add robust finite-value handling:
  - mask singularities,
  - clamp extreme values,
  - preserve singularity metadata.
- Add decimation option for preview.
- Add high-resolution mode for screenshots.
- Add offscreen rendering support for automated gallery generation.

### 1.5 Acceptance criteria

- Existing PyVista examples still work.
- New `SurfaceMesh` API can reproduce current Riemann relief outputs.
- A gallery image can be generated headlessly in CI or local script mode.
- STL export uses the same mesh pipeline as visualization.
- Users can retrieve the underlying PyVista object for custom work.

---

## Phase 2 — Presets, CLI, and Curated Gallery

**Goal:** Make Complexplorer easy to use as a tool, not just as a library.

**Recommended timeframe:** 2–3 weeks  
**Release target:** v2.3.0

### 2.1 Deliverables

- Function preset registry.
- CLI renderer.
- Gallery generator.
- “Function cards” metadata format.
- Curated gallery with canonical functions.

### 2.2 Why this matters

The out-of-the-box ideas depend on a curated library of examples. Games, website interactives, Printables objects, and engineering demos all become easier if functions can be represented as named, metadata-rich presets.

### 2.3 Function preset model

```python
@dataclass
class FunctionPreset:
    id: str
    title: str
    expression: str
    callable: Callable[[np.ndarray], np.ndarray]
    recommended_domain: ComplexDomain
    recommended_cmap: CmapSpec
    singularities: list[SingularityHint]
    story: str
    tags: list[str]
```

Example:

```python
cp.presets.get("pole_flower_10")
cp.presets.get("gamma")
cp.presets.get("sqrt_branch")
cp.presets.get("zeta_critical_strip")
```

### 2.4 CLI examples

```bash
complexplorer render "z/(z**10 - 1)" \
  --domain annulus:0.2:3 \
  --mode riemann-relief \
  --cmap phase \
  --output pole_flower.png
```

```bash
complexplorer stl preset:pole_flower_10 \
  --size-mm 80 \
  --resolution 250 \
  --output pole_flower_10.stl
```

```bash
complexplorer gallery --preset-set canonical --output gallery_output/
```

### 2.5 Initial curated gallery

| Category | Presets |
|---|---|
| Basic maps | `z`, `z^2`, `1/z`, `(z-1)/(z+1)` |
| Singularities | poles of different order, essential singularity `exp(1/z)` |
| Branches | `sqrt(z)`, `log(z)`, `z^(1/3)` |
| Special functions | `Gamma(z)`, `zeta(z)`, `Airy`, `Bessel` |
| Dynamics | Newton basins, rational maps, Julia-like examples |
| Engineering | low-pass filter, notch filter, resonator, impedance semicircle |
| Printable objects | pole flower, zeta medallion, branch shell |

### 2.6 Acceptance criteria

- User can generate a PNG/STL from a CLI command.
- Gallery generation is reproducible.
- Function presets include recommended domains and descriptions.
- Presets are reusable by games and website demos.

---

## Phase 3 — Games and Interactive Learning

**Goal:** Make Complexplorer playful and educational, while creating public-facing website content.

**Recommended timeframe:** 4–8 weeks depending on ambition  
**Release target:** v2.4.0 / v2.5.0

This phase can start as static asset generation before building full web interactivity.

---

# 3A. Singularity Detective

## Concept

Show a complex phase portrait or relief map. The player must identify:

- zeros,
- poles,
- essential singularities,
- branch points,
- and possibly their orders.

This is probably the highest-value game idea.

## Why it works

Complex phase portraits encode mathematical structure. A user can learn to “read” the image:

- zero: phase winds positively around a point;
- pole: phase winds negatively around a point;
- higher-order zero/pole: multiple phase cycles;
- essential singularity: infinitely dense structure nearby;
- branch point: discontinuity or sheet transition.

## MVP

Static levels generated from presets.

```python
level = cp.games.SingularityLevel.from_preset("rational_01")
level.render_prompt("level_01.png")
level.render_solution("level_01_solution.png")
```

Each level produces:

- prompt image,
- solution image,
- JSON answer key,
- markdown explanation.

## Level schema

```json
{
  "id": "singularity_001",
  "title": "Two zeros and one pole",
  "function": "(z - 1) * (z + 0.5) / (z - 0.2j)",
  "domain": {"type": "rectangle", "width": 4, "height": 4},
  "targets": [
    {"type": "zero", "location": [1, 0], "order": 1},
    {"type": "zero", "location": [-0.5, 0], "order": 1},
    {"type": "pole", "location": [0, 0.2], "order": 1}
  ]
}
```

## Advanced version

Interactive web game:

- click to mark points,
- choose type: zero/pole/branch/essential,
- choose order,
- score based on distance and correctness,
- show phase-winding explanation after answer.

## Deliverables

- `complexplorer.games.singularity_detective`
- 20 beginner levels
- 20 intermediate levels
- 10 advanced levels
- static web assets
- markdown explanations

---

# 3B. Branch-Cut Zoo

## Concept

A visual catalog of multi-valued functions and their branch choices.

Functions:

- `sqrt(z)`
- `log(z)`
- `z^(1/3)`
- `arcsin(z)`
- `arccos(z)`
- `LambertW(z)` branches

## User experience

For each function, show:

1. principal branch on the plane,
2. alternative branch cut,
3. Riemann surface / multi-sheet view,
4. path animation around branch point,
5. monodromy result.

## Why it matters

Branch cuts are usually taught as arbitrary ugly scars on the complex plane. Complexplorer can show that the cut is just a local bookkeeping device for a richer global object.

## MVP

Static pages with generated figures.

```bash
complexplorer gallery --preset-set branch-cut-zoo --output docs/gallery/branch_cuts
```

## Advanced version

Interactive path-drawing:

- draw a loop around a branch point,
- show how the function value changes sheet,
- display “you returned / you did not return to original value.”

## Deliverables

- branch-cut preset registry,
- principal and alternate cuts,
- PyVista surface views,
- monodromy path visualizations,
- educational explanations.

---

# 3C. Function Guessr

## Concept

A lightweight game:

> “Which function produced this image?”

Possible choices:

- `z`
- `z^2`
- `1/z`
- `sin(z)`
- `exp(z)`
- `Gamma(z)`
- `sqrt(z)`

## Why it works

This is quick, playful, and shareable. It teaches pattern recognition.

## MVP

- Generate one image.
- Provide four choices.
- Reveal answer with explanation.

## Advanced version

- Daily function puzzle.
- Difficulty levels.
- Shareable score card.
- “What gave it away?” explanation.

---

# 3D. Monodromy Maze

## Concept

A path-based game on a Riemann surface.

Player objective:

> Move around branch points to reach a target sheet/value.

Example:

- Start at a point on the principal sheet of `sqrt(z)`.
- Walk around the origin once.
- You land on the other sheet.
- Walk around again.
- You return.

## Why it is fun

It makes monodromy tactile. Instead of defining it abstractly, the player experiences it.

## MVP

- predefined paths,
- static animation frames,
- simple “which sheet are you on?” prompt.

## Advanced version

- interactive path drawing,
- live sheet index update,
- branch point obstacles,
- target sheet challenge.

---

# 3E. Möbius Playground

## Concept

Interactive visualization of transformations:

```text
f(z) = (a z + b) / (c z + d)
```

Show:

- plane view,
- Riemann sphere view,
- image of circles/lines,
- pole movement,
- fixed points,
- transformation composition.

## Why it fits

Complexplorer already has Riemann sphere machinery. Möbius transformations are one of the most accessible and beautiful demonstrations of complex geometry.

## MVP

- presets for translation, rotation, inversion, dilation,
- side-by-side before/after images,
- Riemann sphere PyVista view.

## Advanced version

- sliders for `a, b, c, d`,
- constrain determinant nonzero,
- show fixed points and circle mappings,
- record animation.

---

## Phase 4 — Riemann Surfaces and Multi-Sheet Geometry

**Goal:** Make Complexplorer capable of visualizing multi-valued complex functions as geometric surfaces.

**Recommended timeframe:** 6–12 weeks  
**Release target:** v3.0.0

This is the major technical and conceptual leap.

---

## 4.1 Scope

Start with finite, explicitly constructible surfaces. Avoid trying to solve the general algebraic-geometry problem immediately.

### MVP surfaces

| Surface | Construction |
|---|---|
| `sqrt(z)` | two sheets stitched across branch cut |
| `z^(1/n)` | n sheets stitched cyclically |
| `sqrt(z-a)` | shifted branch point |
| `sqrt((z-a)(z-b))` | two branch points |
| `log(z)` preview | helical covering visualization, not full infinite surface |

### Phase 4.1 API sketch

```python
surface = cp.surfaces.SqrtSurface(
    branch_point=0,
    cut_angle=np.pi,
    radius=3,
    resolution=300,
)

field = surface.field(lambda z, w: w)
cp.show_surface(surface, field, color_by="phase", height_by="modulus")
```

Alternative ergonomic API:

```python
surface = cp.RiemannSurface.sqrt(domain=cp.Disk(3), branch_point=0)
cp.plot_surface(surface, color_by="w_phase", height_by="abs_w")
```

---

## 4.2 Surface concepts

### `Sheet`

Represents one branch of a multi-valued function.

```python
@dataclass
class Sheet:
    index: int
    z_grid: ComplexGrid
    w_values: np.ndarray
    mask: np.ndarray
```

### `BranchPoint`

```python
@dataclass
class BranchPoint:
    location: complex
    order: int | None
    label: str = ""
```

### `BranchCut`

```python
@dataclass
class BranchCut:
    start: complex
    end: complex | Literal["infinity"]
    sheet_permutation: tuple[int, ...]
```

### `RiemannSurface`

```python
@dataclass
class RiemannSurface:
    sheets: list[Sheet]
    branch_points: list[BranchPoint]
    branch_cuts: list[BranchCut]
    mesh: SurfaceMesh
```

---

## 4.3 Visualization modes

### Sheet stack view

Show sheets separated vertically.

Good for explanation.

### Stitched surface view

Attempt to glue sheets across branch cuts.

Good for conceptual elegance.

### Cutaway view

Show branch cuts and sheet connections with partial transparency.

Good for teaching.

### Path / monodromy view

Draw a path on the base plane and show movement between sheets.

Good for games.

### Projection view

Project surface to the `z` plane, colored by sheet or value.

Good for comparing with ordinary domain coloring.

---

## 4.4 Implementation milestones

### Milestone 4.1 — Explicit two-sheet surfaces

- `sqrt(z)`
- branch point marker
- cut seam visualization
- two-sheet coloring
- PyVista rendering

### Milestone 4.2 — General finite cyclic sheets

- `z^(1/n)`
- sheet index coloring
- cyclic monodromy
- path-around-branch-point demo

### Milestone 4.3 — Two-branch-point surfaces

- `sqrt((z-a)(z-b))`
- stitch between two branch points
- show topological transition from plane cuts to surface handles

### Milestone 4.4 — Algebraic curve preview

Support a small family:

```text
w^2 = P(z)
```

where `P` is a polynomial.

Examples:

```text
w^2 = z
w^2 = z^2 - 1
w^2 = z^3 - z
w^2 = z(z - 1)(z - λ)
```

Do not promise full generality yet.

---

## 4.5 Acceptance criteria

- User can visualize `sqrt(z)` as a two-sheet surface.
- User can visualize `z^(1/3)` as a three-sheet surface.
- User can draw or load a path and see sheet transitions.
- Surface visualizations use PyVista only.
- Surfaces can export screenshots.
- At least one surface can export a printable mesh or object-like STL.

---

## Phase 5 — Electrical Engineering, Filters, and Resonators

**Goal:** Make Complexplorer useful for engineers working with transfer functions, poles, zeros, impedance, filters, RF networks, and resonators.

**Recommended timeframe:** 4–8 weeks  
**Release target:** v3.1.0

This is the most promising “practical application” branch. It connects directly to electrical engineering and can later bleed naturally into physical Objects/Projects.

---

## 5.1 Product thesis for EE mode

Complex-domain engineering is often taught through separate plots:

- pole-zero plot,
- Bode magnitude,
- Bode phase,
- Nyquist plot,
- Smith chart,
- impedance spectrum,
- time response.

Complexplorer can unify some of these by showing the transfer function itself as a complex function:

```text
H(s): complex frequency → complex response
```

or

```text
Z(ω): frequency → complex impedance
```

or

```text
S11(f): frequency → complex reflection coefficient
```

The visual payoff:

> poles, zeros, resonances, notches, and stability margins become visible as geometry.

---

## 5.2 Initial EE module structure

```text
complexplorer/ee/
  transfer_function.py
  filters.py
  resonators.py
  qcm.py
  impedance.py
  rf_network.py
  annotations.py
```

---

## 5.3 Transfer function explorer

### MVP

Support continuous-time transfer functions:

```python
H = cp.ee.TransferFunction(
    numerator=[1],
    denominator=[1, 2, 2],
)

cp.ee.plot_transfer_domain(H, domain="s-plane")
cp.ee.plot_pole_zero(H)
cp.ee.plot_bode(H)
cp.ee.plot_nyquist(H)
```

### Domain coloring view

Plot:

```text
H(s) = N(s) / D(s)
```

over a rectangular region of the complex `s` plane.

Visual encodings:

- phase: `arg(H(s))`
- brightness/height: `|H(s)|`
- markers: poles and zeros
- optional contours: constant gain, constant phase
- stability boundary: imaginary axis

### Recommended demo functions

| Demo | Transfer function |
|---|---|
| first-order low-pass | `1 / (s + 1)` |
| second-order resonator | `ω0² / (s² + 2ζω0s + ω0²)` |
| notch filter | `(s² + ω0²) / (s² + 2ζω0s + ω0²)` |
| unstable pole | `1 / (s - 1)` |
| pole-zero cancellation | `(s + 1) / ((s + 1)(s + 10))` |

---

## 5.4 Filter design visualizer

### Goal

Create educational and practical visualizations for common filters.

Filter families:

- Butterworth,
- Chebyshev I,
- Chebyshev II,
- elliptic,
- Bessel,
- notch,
- comb.

### Views

For each filter:

- pole-zero plot,
- domain coloring of `H(s)` or `H(z)`,
- Bode magnitude/phase,
- group delay,
- optional 3D relief surface of gain.

### API sketch

```python
flt = cp.ee.filters.butterworth(order=5, cutoff=1.0, kind="lowpass")

cp.ee.plot_filter_dashboard(flt)
cp.ee.plot_transfer_domain(flt, plane="s")
cp.ee.plot_filter_relief(flt, modulus_mode="logarithmic")
```

### Out-of-the-box idea

Make a “filter zoo” analogous to the branch-cut zoo:

- Butterworth: maximally flat passband,
- Chebyshev: ripple tradeoff,
- elliptic: aggressive transition,
- Bessel: phase/group-delay behavior.

Show how the pole constellations produce the response.

---

## 5.5 Resonator explorer

### Why this is a strong fit

Resonators are naturally complex:

- frequency response is complex-valued,
- Q factor appears geometrically,
- impedance/admittance traces have meaningful shapes,
- pole locations encode damping,
- magnitude and phase both matter.

### Examples

- RLC series resonator,
- RLC parallel resonator,
- mechanical oscillator analog,
- quartz crystal equivalent circuit,
- QCM-like resonator model.

### API sketch

```python
res = cp.ee.RLCSeries(R=10, L=1e-3, C=1e-9)

cp.ee.plot_impedance(res, f_min=1e3, f_max=10e6)
cp.ee.plot_complex_trace(res)
cp.ee.plot_resonator_relief(res)
```

### Visuals

- impedance trajectory in complex plane,
- admittance trajectory,
- magnitude/phase vs frequency,
- resonance marker,
- half-power bandwidth,
- Q estimate,
- pole-zero view.

---

## 5.6 QCM / quartz resonator branch

This should probably be experimental, but it fits your interests well.

Possible models:

- Butterworth–Van Dyke equivalent circuit,
- motional resistance/inductance/capacitance,
- static capacitance,
- series resonance,
- parallel resonance,
- loaded resonator shifts.

Potential demos:

- how added mass shifts resonance,
- how damping broadens response,
- how viscosity-like loading changes impedance trace,
- how harmonics differ.

Possible API:

```python
qcm = cp.ee.QCMResonator(
    f0=5e6,
    Rm=20,
    Lm=10e-3,
    Cm=0.1e-12,
    C0=5e-12,
)

cp.ee.plot_qcm_response(qcm, harmonics=[1, 3, 5])
cp.ee.plot_impedance_trace(qcm)
cp.ee.plot_resonance_shift(qcm, delta_mass=[0, 10, 20, 50])
```

---

## 5.7 RF / Touchstone / scikit-rf bridge

Do not reimplement RF network analysis from scratch. Use a bridge.

Possible integration:

```python
net = cp.ee.from_touchstone("measurement.s2p")
cp.ee.plot_sparams(net, parameter="S11")
cp.ee.plot_smith_like(net, parameter="S11")
cp.ee.plot_complex_trace(net, parameter="S21")
```

Optional dependency:

```bash
pip install "complexplorer[ee]"
```

This can use `scikit-rf` under the hood for Touchstone and network handling.

---

## 5.8 Acceptance criteria

- User can define a transfer function and get:
  - pole-zero plot,
  - Bode plot,
  - Nyquist plot,
  - domain-colored `s`-plane plot.
- User can generate a second-order resonator visualization.
- User can create at least one printable resonator-inspired object.
- Optional `ee` dependencies are cleanly isolated.
- EE examples are documented in a dedicated tutorial.

---

## Phase 6 — Objects / Projects Integration

**Goal:** Connect Complexplorer outputs to personal-site artifacts, 3D-printable objects, visual essays, and project pages.

**Recommended timeframe:** ongoing after v2.3, major push after v3.0  
**Release target:** not necessarily a package release; this is also website/content work

---

## 6.1 Concept

A Complexplorer “object” is not just an image or STL. It is a packaged artifact with:

- title,
- mathematical definition,
- visual render,
- optional interactive demo,
- optional STL/3MF model,
- print settings,
- explanation,
- source code,
- and a durable project URL.

### Example object card

```yaml
id: pole_flower_10
title: Pole Flower 10
function: z / (z**10 - 1)
type: riemann_relief_object
tags: [complex-analysis, poles, ornament, 3d-printing]
outputs:
  image: pole_flower_10.png
  stl: pole_flower_10.stl
  project_page: /objects/pole-flower-10/
  printables: https://www.printables.com/...
math:
  zeros: [0]
  poles: roots_of_unity_10
  domain: annulus_0p2_3
printing:
  diameter_mm: 80
  nozzle_mm: 0.4
  layer_height_mm: 0.2
  supports: false
```

---

## 6.2 Website taxonomy

Recommended site categories:

```text
/visualizations/
  /complex-functions/
  /riemann-surfaces/
  /engineering/
  /games/

/objects/
  /mathematical-ornaments/
  /function-fossils/
  /resonator-objects/
  /printed-experiments/

/projects/
  /complexplorer/
  /qcm-visualization/
  /filter-zoo/
  /singularity-detective/
```

### Why separate Objects and Projects?

- **Projects** describe ongoing work and systems.
- **Objects** are durable artifacts people can view, print, download, or reference.

Complexplorer can feed both:

- A Riemann-surface implementation is a **project**.
- A printed `sqrt(z)` branch shell is an **object**.
- Singularity Detective is both a **project** and a **game**.

---

## 6.3 Object families

### Function fossils

Physical objects that encode analytic structure.

Examples:

| Object | Function / structure |
|---|---|
| Pole Flower | `z / (z^n - 1)` |
| Branch Shell | `sqrt(z)` |
| Zeta Medallion | `ζ(z)` over selected domain |
| Gamma Terrain | `Γ(z)` relief |
| Essential Singularity Crater | `exp(1/z)` |
| Elliptic Landscape | `w² = z(z-1)(z-λ)` |

### Engineering objects

Physical artifacts inspired by transfer functions and resonators.

Examples:

| Object | Meaning |
|---|---|
| Low-Pass Bowl | first-order pole geometry |
| Notch Filter Saddle | zeros on imaginary axis |
| Resonator Ring | second-order resonance surface |
| QCM Shift Tile | before/after mass-loading response |
| Smith Trace Pendant | measured RF reflection path |

### Game objects

Printable or visual artifacts tied to games.

Examples:

| Object | Game tie-in |
|---|---|
| Singularity Detective Card Set | printed puzzle cards |
| Branch-Cut Maze Tile | monodromy game aid |
| Function Guessr Deck | visual flashcards |

---

## 6.4 Project card export

Add package support for generating website-ready metadata.

```python
card = cp.export.ProjectCard.from_preset(
    "pole_flower_10",
    outputs=["png", "stl", "markdown", "json"],
)
card.write("site/content/objects/pole-flower-10/")
```

Generated files:

```text
pole-flower-10/
  index.md
  metadata.json
  pole_flower_10.png
  pole_flower_10.stl
  source.py
```

---

## Phase 7 — Advanced Mathematical Directions

**Goal:** Build a long-term backlog of mathematically rich directions without derailing the near-term architecture.

These are not all immediate implementation items. Treat them as a research backlog and source of future visual essays.

---

## 7.1 Special Function Atlas

Curated visual atlas of:

- Gamma function,
- Riemann zeta,
- Airy functions,
- Bessel functions,
- elliptic functions,
- theta functions,
- modular forms.

Deliverables:

- gallery pages,
- function cards,
- printable highlights,
- educational essays.

---

## 7.2 Hyperbolic Domain Coloring

Explore complex functions on:

- Poincaré disk,
- upper half-plane,
- modular fundamental domains,
- hyperbolic tilings.

This could produce striking website demos and connect naturally to modular forms.

---

## 7.3 Complex Dynamics Aquarium

Interactive zoo of:

- Newton fractals,
- Julia sets,
- Mandelbrot-related maps,
- rational map dynamics,
- basins of attraction.

Complexplorer should not become “yet another fractal generator,” but dynamics can be a strong visual branch if tied to phase portraits and function behavior.

---

## 7.4 Conformal Map Lab

Show how domains transform:

- grid deformation,
- boundary mapping,
- angle preservation,
- singularity behavior,
- inverse maps.

Potential demos:

- disk to half-plane,
- strip to annulus,
- Joukowski airfoil map,
- Schwarz–Christoffel-inspired examples.

---

## 7.5 Algebraic Curve Surface Lab

Long-term goal:

```python
surface = cp.RiemannSurface.from_curve("w**2 = z**3 - z")
```

This is hard, but very compelling.

Start with constrained polynomial families before general symbolic parsing.

---

## 8. Detailed Version Roadmap

## v2.1 — Stabilize and clarify

**Theme:** polish the existing package.

Deliverables:

- metadata/license cleanup,
- backend policy docs,
- formatter/linter,
- initial CLI skeleton,
- current-gallery cleanup,
- “PyVista is primary 3D backend” documentation.

Suggested release notes:

> v2.1 clarifies Complexplorer’s backend strategy and prepares the package for the next generation of PyVista-first 3D visualizations.

---

## v2.2 — PyVista 3D kernel

**Theme:** consolidate 3D internals.

Deliverables:

- `plot3d` package,
- `SurfaceMesh` abstraction,
- shared mesh pipeline,
- camera and lighting presets,
- screenshot/export helpers,
- old PyVista functions migrated to shared internals.

---

## v2.3 — CLI and gallery engine

**Theme:** make it a tool.

Deliverables:

- `complexplorer render`,
- `complexplorer stl`,
- `complexplorer gallery`,
- function preset registry,
- curated canonical gallery,
- project-card metadata prototype.

---

## v2.4 — Games MVP

**Theme:** make it playful.

Deliverables:

- Singularity Detective static levels,
- Function Guessr static levels,
- Branch-Cut Zoo static gallery,
- generated markdown explanations,
- web-ready assets.

---

## v2.5 — Engineering MVP

**Theme:** make it useful.

Deliverables:

- transfer function class,
- poles/zeros annotations,
- Bode/Nyquist helpers,
- domain-colored `s`-plane view,
- resonator examples,
- first filter zoo examples.

---

## v3.0 — Riemann surfaces

**Theme:** multi-sheet complex geometry.

Deliverables:

- PyVista required dependency,
- explicit finite-sheet surfaces,
- `sqrt(z)` surface,
- `z^(1/n)` surfaces,
- branch point / branch cut objects,
- monodromy path visualization,
- Riemann surface tutorial,
- at least one printable branch-surface object.

---

## v3.1 — EE / resonator expansion

**Theme:** practical engineering applications.

Deliverables:

- filter zoo,
- resonator explorer,
- optional scikit-rf bridge,
- Touchstone import prototype,
- QCM resonator demo,
- engineering object exports.

---

## 9. API Design Examples

## 9.1 Simple 2D phase portrait

```python
import complexplorer as cp

f = lambda z: (z**2 - 1) / (z**2 + 1)
domain = cp.Rectangle(4, 4)

cp.plot(domain, f, cmap=cp.Phase(phase_sectors=12, auto_scale_r=True))
```

No major change needed.

---

## 9.2 PyVista relief map

```python
field = cp.sample(f, domain, resolution=800)
mesh = cp.mesh.riemann_relief(field, modulus_mode="arctan")

cp.show(mesh, camera="isometric", lighting="studio")
```

---

## 9.3 Export object

```python
artifact = cp.objects.FunctionObject.from_function(
    f,
    domain=cp.Annulus(0.2, 3.0),
    title="Pole Flower 10",
    modulus_mode="arctan",
)

artifact.render("pole_flower_10.png")
artifact.export_stl("pole_flower_10.stl", size_mm=80)
artifact.write_project_card("site/objects/pole-flower-10")
```

---

## 9.4 Riemann surface

```python
surface = cp.RiemannSurface.sqrt(
    domain=cp.Disk(3),
    branch_point=0,
    cut_angle=np.pi,
    sheets=2,
)

cp.plot_surface(
    surface,
    color_by="phase_w",
    height_by="abs_w",
    view="stitched",
)
```

---

## 9.5 Monodromy path

```python
surface = cp.RiemannSurface.sqrt(domain=cp.Disk(3))
path = cp.Path.circle(center=0, radius=1.0, turns=1)

result = surface.follow(path, start_sheet=0)

cp.plot_monodromy(surface, path, result)
print(result.end_sheet)
```

Expected result for `sqrt(z)`: after one loop around the origin, the path lands on the other sheet.

---

## 9.6 Transfer function

```python
H = cp.ee.TransferFunction(
    numerator=[1],
    denominator=[1, 0.2, 1],
)

cp.ee.plot_domain_coloring(H, plane="s", bounds=(-2, 1, -3, 3))
cp.ee.plot_pole_zero(H)
cp.ee.plot_bode(H)
cp.ee.plot_nyquist(H)
```

---

## 9.7 Resonator object

```python
res = cp.ee.RLCSeries(R=10, L=1e-3, C=1e-9)

obj = cp.objects.EngineeringObject.from_resonator(
    res,
    title="Series RLC Resonator Relief",
    f_min=1e3,
    f_max=10e6,
)

obj.render("rlc_resonator.png")
obj.export_stl("rlc_resonator.stl")
```

---

## 10. Testing Strategy

## 10.1 Unit tests

Core tests:

- domain masks,
- grid sampling,
- colormap output shape/range,
- modulus scaling,
- singularity handling,
- preset loading,
- CLI argument parsing.

## 10.2 Numerical tests

Examples:

- zeros/poles detected near known locations,
- phase winding around simple zero is approximately `+1`,
- phase winding around simple pole is approximately `-1`,
- `sqrt(z)` sheet transition works after one loop,
- transfer-function poles/zeros match polynomial roots,
- Bode data matches SciPy/control output.

## 10.3 Mesh tests

For PyVista/3D:

- mesh has finite vertices,
- mesh has faces,
- scalar fields have correct length,
- STL export produces non-empty file,
- optional watertightness check for printable objects,
- no NaNs in exported mesh.

## 10.4 Image regression tests

Use lightweight snapshot tests for:

- canonical 2D phase portrait,
- canonical Riemann relief screenshot,
- branch-cut zoo example,
- transfer function domain coloring.

Do not make image tests too brittle. Use perceptual or tolerance-based tests if possible.

## 10.5 Performance tests

Track:

- grid sampling time,
- PyVista mesh construction time,
- screenshot generation time,
- STL export time,
- memory usage for large grids.

Suggested benchmark cases:

| Case | Resolution | Purpose |
|---|---:|---|
| small preview | 200 | fast dev loop |
| normal gallery | 800 | high-quality images |
| print mesh | 250 | STL generation |
| stress | 1500 | performance boundary |

---

## 11. Documentation Plan

## 11.1 Docs structure

```text
docs/
  index.md
  installation.md
  quickstart.md
  user-guide/
    2d-phase-portraits.md
    colormaps.md
    domains.md
    pyvista-3d.md
    riemann-relief.md
    stl-export.md
    riemann-surfaces.md
    electrical-engineering.md
    games.md
  gallery/
    canonical-functions.md
    branch-cut-zoo.md
    filter-zoo.md
    objects.md
  api/
    core.md
    plot2d.md
    plot3d.md
    surfaces.md
    ee.md
    games.md
    export.md
  development/
    architecture.md
    backend-policy.md
    roadmap.md
```

## 11.2 Key tutorials

### Tutorial 1 — “Read a phase portrait”

Teach:

- phase hue,
- modulus brightness/height,
- zeros,
- poles,
- winding.

### Tutorial 2 — “From function to object”

Teach:

- domain selection,
- relief map,
- scaling mode,
- mesh preview,
- STL export,
- print considerations.

### Tutorial 3 — “Branch cuts are seams, not scars”

Teach:

- principal branch,
- alternative branch cuts,
- Riemann surface view,
- monodromy.

### Tutorial 4 — “Transfer functions as complex landscapes”

Teach:

- poles and zeros,
- Bode/Nyquist connection,
- domain coloring of `H(s)`,
- resonator example.

### Tutorial 5 — “Build a puzzle level”

Teach:

- generate a Singularity Detective level,
- export prompt/solution,
- write explanation.

---

## 12. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| PyVista dependency feels heavy | Some users may prefer lightweight 2D | Stage transition; consider future `complexplorer-lite`; explain clearly |
| Riemann surfaces become too general too quickly | Project gets bogged down | Start with explicit surfaces, not general algebraic solver |
| Games distract from math library | Scope creep | Use games as generated assets from core primitives |
| EE module becomes a full controls/RF package | Reinventing existing libraries | Use SciPy/control/scikit-rf bridges; Complexplorer owns visualization |
| STL export creates unprintable meshes | Bad physical artifacts | Add printability checks, constraints, and docs |
| Gallery generation becomes brittle | Docs break often | Use presets and deterministic rendering settings |
| Too many APIs | User confusion | Keep high-level functions ergonomic; advanced objects optional |

---

## 13. Design Principles

### 13.1 Make mathematical structure explicit

Do not just render arrays. Preserve concepts:

- zero,
- pole,
- branch point,
- sheet,
- cut,
- path,
- monodromy,
- transfer function,
- resonance.

### 13.2 Keep 2D and 3D responsibilities separate

2D:

- clean,
- printable,
- publication-friendly,
- matplotlib-based.

3D:

- interactive,
- mesh-oriented,
- PyVista-based,
- exportable.

### 13.3 Prefer presets over one-off examples

A good preset can power:

- a gallery page,
- a game level,
- an STL object,
- a website project card,
- a tutorial.

### 13.4 Avoid fake generality

It is better to support `sqrt(z)`, `z^(1/n)`, and `w² = P(z)` very well than to expose a grand API for arbitrary Riemann surfaces that fails on hard cases.

### 13.5 Treat objects as first-class outputs

Images and meshes should carry metadata:

- function,
- domain,
- scaling,
- colormap,
- version,
- generation script,
- print settings.

This makes objects reproducible and website-friendly.

---

## 14. Prioritized Backlog

## Must do next

1. Fix license metadata mismatch.
2. Add backend policy docs.
3. Add CLI skeleton.
4. Refactor PyVista 3D functions into shared internals.
5. Add function preset registry.
6. Add gallery generator.
7. Add Singularity Detective static MVP.
8. Add transfer-function MVP.
9. Add explicit `sqrt(z)` Riemann surface.

## Should do soon

1. Add object/project card export.
2. Add Branch-Cut Zoo.
3. Add Möbius Playground static version.
4. Add filter zoo.
5. Add resonator explorer.
6. Add PyVista camera/lighting presets.
7. Add mesh validation tests.

## Could do later

1. Touchstone import.
2. QCM equivalent-circuit demos.
3. Hyperbolic tiling experiments.
4. Algebraic curve family `w² = P(z)`.
5. Function Guessr daily puzzle.
6. Monodromy Maze interactive game.
7. glTF export.
8. animated videos.

## Avoid for now

1. General symbolic algebraic-curve solver.
2. Full browser-based live rendering engine.
3. Full RF simulator.
4. Full controls package.
5. Heavy web app before static assets prove value.
6. Maintaining advanced matplotlib 3D parity.

---

## 15. Suggested 90-Day Plan

## Month 1 — Strengthen the foundation

Target:

- v2.1 / v2.2 foundation.

Work:

- metadata/license cleanup,
- formatting/linting,
- backend policy,
- PyVista 3D kernel,
- mesh wrapper,
- camera/lighting presets,
- CLI skeleton.

Outcome:

> Complexplorer becomes easier to extend and clearly PyVista-first for 3D.

---

## Month 2 — Make it visible and playful

Target:

- v2.3 / v2.4 demos.

Work:

- function preset registry,
- gallery generator,
- canonical gallery,
- Singularity Detective MVP,
- Branch-Cut Zoo static pages,
- Function Guessr prototype,
- project-card metadata prototype.

Outcome:

> Complexplorer produces website-ready assets, game levels, and curated demos.

---

## Month 3 — Add depth: EE and Riemann surfaces

Target:

- v2.5 prototype + v3.0 groundwork.

Work:

- transfer-function visualizer,
- pole-zero annotations,
- Bode/Nyquist companion plots,
- resonator examples,
- `sqrt(z)` two-sheet surface,
- monodromy path demo,
- first printable branch object.

Outcome:

> Complexplorer now has two powerful expansion branches: engineering visualization and multi-sheet complex geometry.

---

## 16. Concrete First Five Issues to Open

### Issue 1 — Clarify backend policy and PyVista roadmap

**Title:** Define 2D/3D backend policy and PyVista-first roadmap

**Description:**

Document that matplotlib remains the 2D backend, while PyVista is the primary and eventually required backend for advanced 3D visualization, Riemann relief maps, surfaces, and mesh export.

**Deliverables:**

- `docs/development/backend-policy.md`
- README note
- migration note for v3.0

---

### Issue 2 — Add `SurfaceMesh` wrapper

**Title:** Add `SurfaceMesh` abstraction for PyVista-backed 3D outputs

**Description:**

Introduce a wrapper around PyVista meshes that stores mesh data, scalar fields, color fields, metadata, and export methods.

**Deliverables:**

- `complexplorer/plot3d/surface_mesh.py`
- tests for mesh creation
- support for `.to_pyvista()`
- support for screenshot/STL export

---

### Issue 3 — Add function preset registry

**Title:** Add reusable function preset registry

**Description:**

Create metadata-rich function presets that can power docs, gallery generation, games, CLI rendering, and object export.

**Deliverables:**

- `complexplorer/core/presets.py`
- 20 canonical presets
- tests
- docs page

---

### Issue 4 — Add CLI renderer

**Title:** Add `complexplorer render` and `complexplorer stl`

**Description:**

Provide command-line access to common rendering and export workflows.

**Deliverables:**

- `complexplorer/cli/main.py`
- `complexplorer render ...`
- `complexplorer stl ...`
- docs examples

---

### Issue 5 — Singularity Detective MVP

**Title:** Generate static Singularity Detective levels

**Description:**

Create a game asset generator for phase-portrait puzzles involving zeros, poles, branch points, and essential singularities.

**Deliverables:**

- `complexplorer/games/singularity_detective.py`
- level schema
- 10 beginner levels
- prompt image export
- solution image export
- markdown explanation export

---

## 17. Recommended First Demo Set

If you want a quick public-facing win, build these in order:

1. **Pole Flower Object**  
   `f(z) = z / (z^10 - 1)`  
   Output: PNG, STL, project card.

2. **Singularity Detective Level 1**  
   Simple rational function with two zeros and one pole.  
   Output: prompt, solution, explanation.

3. **Branch-Cut Zoo: sqrt(z)**  
   Principal branch, alternate cut, two-sheet PyVista view.  
   Output: gallery page.

4. **Second-Order Resonator**  
   Transfer function domain coloring + Bode/Nyquist.  
   Output: engineering demo page.

5. **Möbius Playground Static Demo**  
   Translation, inversion, rotation, sphere view.  
   Output: visual essay.

This set covers all major themes:

- math,
- games,
- physical object,
- Riemann surface,
- engineering application,
- website growth.

---

## 18. Final Recommendation

The best next move is not to immediately chase the hardest Riemann-surface implementation. The best move is to build the **infrastructure that makes many future directions cheap**:

1. PyVista-first 3D kernel.
2. Function preset registry.
3. CLI/gallery generation.
4. Project/object metadata export.
5. One game MVP.
6. One engineering MVP.
7. One Riemann-surface MVP.

That combination will make Complexplorer feel larger than a plotting package.

It will become a platform for:

- mathematical visualization,
- playful learning,
- engineering intuition,
- printable artifacts,
- and personal-site projects.

The highest-leverage tagline remains:

> **Complexplorer makes complex functions visible, explorable, and physical.**

---

## 19. References and useful external anchors

- Complexplorer repository: <https://github.com/kuvychko/complexplorer>
- Complexplorer documentation: <https://kuvychko.github.io/complexplorer/>
- PyVista: <https://pyvista.org/>
- SciPy signal LTI documentation: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html>
- scikit-rf: <https://scikit-rf.org/>
