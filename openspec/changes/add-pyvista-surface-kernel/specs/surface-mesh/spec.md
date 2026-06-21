# Surface Mesh Kernel

## ADDED Requirements

### Requirement: Sampled complex field container

The library SHALL provide a backend-agnostic `ComplexField` produced by sampling a complex
function over a domain (planar) or over the sphere, holding the sample coordinates, the
complex values `w = f(z)`, modulus `|w|`, phase `arg w`, an out-of-domain mask, and
metadata. It SHALL NOT require PyVista to be importable.

#### Scenario: Sample a function over a planar domain

- **WHEN** `sample` is called with a function and a domain at a given resolution
- **THEN** a `ComplexField` is returned whose values equal `f` evaluated on the domain
  mesh, with modulus, phase, and the domain mask populated

#### Scenario: Infinities and NaNs are handled once

- **WHEN** the sampled function produces infinite or NaN values
- **THEN** the field resolves them consistently (finite substitution / masking) so that
  downstream mesh builders receive well-defined moduli

### Requirement: Unified surface mesh object

The library SHALL provide a `SurfaceMesh` that wraps a PyVista dataset together with
attached color/scalar fields and metadata, and exposes a single path to decorate and
export it. It SHALL expose the underlying PyVista object for advanced use.

#### Scenario: Attach colors and scalars

- **WHEN** a `SurfaceMesh` is decorated with a colormap
- **THEN** it carries an `RGB` field plus `magnitude` and `phase` scalars of the correct
  length, produced by a single shared code path regardless of mesh topology

#### Scenario: Access the underlying PyVista mesh

- **WHEN** `to_pyvista()` is called on a `SurfaceMesh`
- **THEN** the underlying `pyvista` dataset is returned for custom work

#### Scenario: Export to STL from any surface

- **WHEN** `save_stl()` is called on a `SurfaceMesh` (landscape or relief)
- **THEN** the mesh is triangulated and written as a non-empty STL file containing no NaN
  vertices

### Requirement: Topology builders share scaling and decoration

The library SHALL build both planar landscape meshes (height derived from `|f|`) and
relief meshes (radius derived from `|f|`) from a `ComplexField`, using the shared modulus
scaling kernel, and returning `SurfaceMesh` objects that share one decoration/export path.

#### Scenario: Landscape builder

- **WHEN** a landscape mesh is built from a field
- **THEN** vertex height is the scaled modulus (with height cap and logarithmic options
  applied as requested) and the result is a `SurfaceMesh`

#### Scenario: Relief builder

- **WHEN** a relief mesh is built from a field
- **THEN** vertex radius is the scaled modulus (bounded by `r_min`/`r_max`) and the result
  is a `SurfaceMesh`

### Requirement: Existing PyVista entry points preserved

The existing public PyVista functions SHALL keep their signatures and behavior (delegating
to the kernel), with the sole intentional exception that `riemann_pv`'s sphere orientation
is corrected to the canonical projection convention (see the `riemann-sphere` capability).

#### Scenario: Landscape outputs unchanged

- **WHEN** `plot_landscape_pv` or `pair_plot_landscape_pv` is called as before
- **THEN** the produced mesh geometry, colors, and scalars match the pre-refactor output

#### Scenario: Riemann sphere output changes only in orientation

- **WHEN** `riemann_pv` is called as before
- **THEN** its mesh colors, scalars, and modulus relief match the pre-refactor output,
  differing only by the corrected projection orientation (`z = 0` now at the south pole)
