## MODIFIED Requirements

### Requirement: The gallery showcases the colormap family

Because the curated presets all use `Phase`, the showcase SHALL additionally render a colormap
gallery: one designated reference function (a catalog preset) rendered under each of the
library's public colormaps. The set comprises:

- `Phase` and its enhancement parameter variants
- `Chessboard`
- `PolarChessboard` (linear and log spacing)
- `LogRings`
- each perceptual phase-portrait colormap (`OklabPhase`, `PerceptualPastel`, `AnalogousWedge`,
  `DivergingWarmCool`, `Isoluminant`, `CubehelixPhase`, `InkPaper`, `EarthTopographic`,
  `FourQuadrant`)

These renders SHALL be written under a reserved `_colormaps/` directory (`_colormaps/<name>.png`),
recorded in `showcase.json` under a `colormaps` section (with the reference preset id), and
presented as a colormap section in the generated docs gallery. The colormap gallery SHALL cover
exactly the colormaps that exist in the public API. It SHALL NOT reference unimplemented
colormaps, and it SHALL NOT omit a public one.

#### Scenario: A colormap gallery is produced for the reference function

- **WHEN** the showcase runs
- **THEN** the designated reference function is rendered under each public colormap, each
  written to `_colormaps/<name>.png` and listed under `showcase.json`'s `colormaps` section with
  the reference preset id

#### Scenario: Every public colormap appears in the colormap gallery

- **WHEN** the `colormaps` section of `showcase.json` is compared with the colormap classes exported by `complexplorer`
- **THEN** every exported concrete colormap class is rendered at least once, and no entry names a class that is not exported

#### Scenario: Colormap-section snippets reconstruct the colormap explicitly

- **WHEN** a colormap-section entry's snippet is generated
- **THEN** it uses the registry reference function (`cp.catalog.get(<id>)`) and constructs the
  colormap explicitly (e.g. `cp.Chessboard(spacing=0.25)`, `cp.Phase(phase_sectors=6)`), runs as
  shown, and references only 3.0-surface APIs

### Requirement: Notebooks cover the 3.0 feature surface

The tutorials SHALL cover the headline 3.0 additions:

- a Riemann-**surface** example (`riemann_surface_pv`)
- the preset registry / gallery workflow (`cp.catalog`, and a pointer to the gallery producer)

Colormap material SHALL cover the library's public colormaps, including the perceptual family,
and SHALL reference only colormaps that exist in the public API. At least one tutorial SHALL
address color-vision deficiency (CVD): which colormaps remain interpretable under common CVD
types, and how to choose one.

#### Scenario: Riemann surfaces and the registry are demonstrated

- **WHEN** the tutorial set is read
- **THEN** at least one notebook demonstrates `riemann_surface_pv` and at least one demonstrates
  the preset registry (`cp.catalog`) with a pointer to the gallery producer

#### Scenario: Color-vision guidance is present

- **WHEN** the tutorial set is read
- **THEN** at least one notebook compares colormaps under simulated color-vision deficiency and recommends colormaps that remain interpretable

#### Scenario: Colormap material matches the public API

- **WHEN** the notebook sources are scanned for colormap class names
- **THEN** every referenced colormap is exported by `complexplorer`, and phase-sector counts are passed as `phase_sectors` (never `n_phi`)
