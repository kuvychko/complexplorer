# STL Export

## ADDED Requirements

### Requirement: Ornament orientation matches the rendered sphere

The Riemann-sphere STL ornament SHALL use the same stereographic projection convention as
the on-screen Riemann sphere visualization (`riemann_pv`), so that a printed ornament
matches the rendered sphere rather than being mirrored. The library SHALL use a single
canonical stereographic projection function for both paths.

#### Scenario: Printed orientation matches the visualization

- **WHEN** an ornament is generated for a function and the same function is rendered with
  `riemann_pv`
- **THEN** the ornament's orientation (which pole corresponds to `z = 0` versus `∞`)
  matches the rendered sphere, not its mirror image

#### Scenario: Single projection implementation

- **WHEN** stereographic projection is performed for visualization or for STL export
- **THEN** both use one canonical projection function (no divergent duplicate
  implementations)
