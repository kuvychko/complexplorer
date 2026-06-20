# STL Export

## Purpose

The stl-export capability turns a complex function into a 3D-printable mathematical ornament: a
modulus-scaled Riemann sphere whose surface relief encodes `|f(z)|`, exported as an STL file sized
in millimeters and repaired for printability. It is built on PyVista and is the bridge from
mathematics to a physical object.

## Requirements

### Requirement: Optional-dependency gating

STL export SHALL be available only when PyVista is installed and SHALL fail with a clear error
otherwise, without breaking import of the rest of the library.

#### Scenario: Missing PyVista raises a clear error

- **WHEN** an ornament generator is constructed and PyVista is not installed
- **THEN** an `ImportError` explaining that PyVista is required (with install guidance) is raised

### Requirement: Ornament mesh generation

The library SHALL generate a 3D mesh from a complex function by distorting a Riemann sphere
radially according to `|f(z)|` and attaching per-point color, magnitude, phase, and radius data.

#### Scenario: Generate the ornament mesh

- **WHEN** an ornament's mesh is generated for a function at a resolution and modulus mode
- **THEN** a sphere mesh is produced whose radius is scaled by the modulus mode applied to `|f(z)|`, carrying RGB color, magnitude, phase, and radius arrays

#### Scenario: Saving before generation is rejected

- **WHEN** validation or saving is requested before the mesh has been generated
- **THEN** an error is raised directing the caller to generate the mesh first

### Requirement: Printability repair and validation

The library SHALL repair and validate the ornament mesh for 3D printing, attempting to make it
watertight and reporting topology, dimensions, and wall-thickness checks, while tolerating the
small polar gaps inherent to a rectangular sphere mesh.

#### Scenario: Repair attempts to close holes

- **WHEN** the mesh is saved with repair enabled
- **THEN** cleaning and hole-filling are applied to make the mesh as watertight as possible before export

#### Scenario: Small residual gaps are tolerated

- **WHEN** validation finds a small number of boundary edges typical of a Riemann sphere
- **THEN** the mesh is reported as printable with a note rather than treated as a hard failure

#### Scenario: Wall thickness is checked against print size

- **WHEN** validation runs for a given target print size
- **THEN** an estimated minimum wall thickness is computed and compared against the printable minimum, recommending a larger size if too thin

### Requirement: Sized STL file output

The library SHALL write the ornament to an STL file centered at the origin and uniformly scaled so
its largest dimension equals a requested size in millimeters, in binary or ASCII form.

#### Scenario: Export at a target size

- **WHEN** the ornament is saved with a target size in millimeters
- **THEN** the mesh is centered, scaled so its maximum dimension equals that size, and written to the STL path, creating the output directory if needed

#### Scenario: One-call generate-and-save

- **WHEN** the combined generate-and-save entry point (or the `create_ornament` convenience function) is called
- **THEN** the mesh is generated and then exported in a single step, returning the saved file path

### Requirement: Non-destructive operations

Mesh operations that scale, center, or repair SHALL operate on copies so the generator's internal
mesh is not mutated by an export.

#### Scenario: Export leaves the source mesh intact

- **WHEN** an ornament is saved
- **THEN** the centering and scaling apply to a copy, and the generator's stored mesh is unchanged for reuse
