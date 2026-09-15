## ADDED Requirements

### Requirement: Renders use one locked profile per family

The showcase producer SHALL define one render profile per family (2D portrait, analytic landscape,
Riemann sphere, relief, Riemann surface, STL mesh) and SHALL apply it to every render of that
family, covering ground, lighting, antialiasing, orientation widget, camera framing, window size
and mesh resolution. Profiles SHALL be declared in one place rather than passed ad-hoc at each
call site, so the committed gallery shares a single look.

#### Scenario: Every render of a family uses that family's profile

- **WHEN** the producer renders any asset, whether registry-driven or curated
- **THEN** it takes the ground, lighting, antialiasing, widget, framing, window size and mesh resolution from that family's profile entry

#### Scenario: The profile is recorded with the asset

- **WHEN** a render is recorded in `showcase.json`
- **THEN** the record names the profile and the mesh resolution used, so the settings behind any committed image are recoverable

### Requirement: A curated tour demonstrates the capabilities the registry cannot express

Beyond the tag-driven preset renders, the producer SHALL render a curated tour section covering
the capabilities that are not catalog presets. The tour SHALL include at least:

- a 2D portrait with the phase-wheel legend enabled
- the same function as an analytic landscape, so the 2D to 3D relationship is visible
- an engineering-mode figure (transfer portrait with poles, zeros and the stability boundary, plus
  the Bode and Nyquist views)
- a composition proof: one `TransferFunction` rendered by `cp.ee` and by a general renderer
- a portrait whose geometry depends on a composite domain (union, intersection or difference)
- a Riemann sphere and a Riemann surface presented side by side
- the physical-output story: a relief render, its STL mesh, and a printed object

Each tour asset SHALL be recorded with the inputs that produced it (function, domain, colormap,
scaling, profile and resolution), a one- or two-sentence interpretation, alt text naming the render
type, and a runnable snippet.

#### Scenario: The tour covers the rev3 capability set

- **WHEN** the tour section of `showcase.json` is read
- **THEN** it contains an entry for each capability listed above, and every referenced image file exists

#### Scenario: Each tour entry carries its recipe and prose

- **WHEN** a tour entry is inspected
- **THEN** it records the inputs that produced the image, a caption, alt text that distinguishes the render type, and a snippet

#### Scenario: Snippets reference only the public API

- **WHEN** the snippets in the tour and colormap sections are parsed
- **THEN** each one parses as Python, and every `complexplorer` name it references is exported by the package (or by `cp.ee`)

### Requirement: Curated assets are recorded, never regenerated

Assets that cannot be produced by running the producer — photographs of printed objects, and any
hand-made image — SHALL be recorded in `showcase.json` as curated, with their kind and origin, and
SHALL be exempt from the regenerable invariant. A curated entry SHALL NOT claim a render recipe,
and the producer SHALL NOT overwrite one.

#### Scenario: A photograph is recorded as curated

- **WHEN** a photograph of a printed ornament is added to the gallery
- **THEN** it is listed in the curated section with its kind and origin, and no code path regenerates it

#### Scenario: Curated and generated assets are distinguishable

- **WHEN** the manifest is read
- **THEN** every image is either a render with a recipe or a curated asset with an origin, and none is both

### Requirement: A hero montage summarises the release

The producer SHALL generate a hero montage laying out six panels — domain coloring, analytic
landscape, Riemann sphere or relief, multivalued surface, engineering mode and physical output —
with consistent crops, labels and dimensions, at a size that stays legible at typical README width.
The montage SHALL be generated from already-rendered assets, so it reproduces from the same
command. When no photograph is available, the physical-output panel SHALL fall back to the STL mesh
render rather than failing.

#### Scenario: The montage is generated and recorded

- **WHEN** the producer runs
- **THEN** the montage image is written, recorded in `showcase.json`, and lists the assets it composed

#### Scenario: The montage degrades without a photograph

- **WHEN** no curated photograph is present
- **THEN** the montage still renders, using the mesh render for the physical-output panel

### Requirement: The gallery page is navigable

The generated gallery page SHALL open with a thumbnail grid or table of contents linking to its
sections, SHALL group entries by idea (phase portraits; mapping and topology; Riemann surfaces;
engineering; colormaps; physical output) rather than by registry order, and SHALL present each
figure with an interpretation and alt text that names the render type. Thumbnails SHALL link to the
full-resolution image. Code snippets SHALL be present but visually secondary. The page SHALL state
which file to edit and which command regenerates it.

#### Scenario: The page opens with navigation

- **WHEN** the generated gallery page is read
- **THEN** it begins with a thumbnail grid or table of contents linking to each section, followed by the idea-ordered sections

#### Scenario: Figures carry interpretation and useful alt text

- **WHEN** any figure on the page is inspected
- **THEN** it has a caption explaining what structure is visible, and alt text that names the render type rather than repeating the title alone

#### Scenario: The page says how it was made

- **WHEN** a reader wants to change the page
- **THEN** the page names the producer file and the command that regenerates it

### Requirement: The producer can render into a staging directory

The producer SHALL accept an output directory and a section selector, so a render can be produced
for review without touching the committed gallery. Committed assets SHALL be updated only by
running the producer against the committed tree.

#### Scenario: Staging output leaves the committed gallery untouched

- **WHEN** the producer is run with an output directory outside the repository gallery
- **THEN** every image, manifest and page it writes goes to that directory, and the committed gallery is unchanged

#### Scenario: A single section can be regenerated

- **WHEN** the producer is run with a section selector
- **THEN** only that section is re-rendered, and the other sections' committed records are preserved

## MODIFIED Requirements

### Requirement: The visual gallery is committed but regenerable from one command

The registry-driven gallery renders SHALL be committed to the repository (so GitHub and PyPI render
them) and SHALL be reproducible by running the showcase producer once
(`python examples/showcase.py`). Every committed *render* SHALL be id-addressed
(`<id>/<render-type>.png`) or written under a reserved section the producer emits (`_colormaps/`,
`_tour/`, thumbnails, the hero montage). Curated assets (photographs and the hero banner) MAY be
hand-maintained and exempt from the regenerable invariant, provided each is recorded in
`showcase.json` as curated; no OTHER hand-named legacy image SHALL remain. STL meshes SHALL NOT be
committed — the gallery shows a relief render and links the generation code instead.

#### Scenario: Every render is reproducible; only recorded curated assets are exempt

- **WHEN** the committed `examples/gallery/` tree is inspected
- **THEN** every image is either a render the producer emits (id-addressed, or under a reserved
  section directory) or an entry in the manifest's curated section, and no other hand-named legacy
  image remains

#### Scenario: No STL binaries are committed

- **WHEN** the gallery tree is inspected
- **THEN** it contains no `.stl` files; ornament presets are represented by a relief PNG plus a
  link to the generation code
