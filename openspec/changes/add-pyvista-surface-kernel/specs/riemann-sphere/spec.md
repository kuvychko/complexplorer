# Riemann Sphere

## ADDED Requirements

### Requirement: Canonical stereographic projection convention

All sphere renderers and the STL ornament SHALL use one canonical stereographic projection
convention via a single shared projection implementation — the documented core convention
in which `z = 0` maps to the south pole and `z = ∞` to the north pole — so that the same
function yields the same orientation across matplotlib `riemann`, PyVista `riemann_pv`, and
the exported ornament, and the printed object matches the rendered sphere.

#### Scenario: All sphere outputs share one orientation

- **WHEN** the same function is rendered by `riemann`, by `riemann_pv`, and exported as an
  STL ornament
- **THEN** all three place `z = 0` at the same pole (south) and are not mirror images of
  one another

#### Scenario: Single projection implementation

- **WHEN** a sphere point is mapped to a complex value for any sphere renderer or for STL
  export
- **THEN** the mapping uses one canonical stereographic projection function (no divergent
  duplicate implementations)

## MODIFIED Requirements

### Requirement: Backend parity with documented differences

The matplotlib and PyVista Riemann renderers SHALL present the same conceptual
visualization, while each MAY differ in mesh construction, default resolution,
interactivity, and grid overlays. They SHALL NOT differ in projection orientation: both use
the canonical convention (`z = 0` at the south pole).

#### Scenario: Both backends render the same concept

- **WHEN** the same function and modulus mode are rendered by the matplotlib and PyVista Riemann renderers
- **THEN** both show a phase-colored sphere with equivalent modulus relief and the same orientation, differing only in performance, interactivity, and backend-specific options (such as the PyVista latitude/longitude grid overlay)
