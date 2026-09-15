## MODIFIED Requirements

### Requirement: Region membership testing

Every domain SHALL classify any array of complex points as inside or outside the region,
with boundary points treated as inside. Membership SHALL be determined by the domain's own
defining parameters (a `Rectangle` by its `re_length`/`im_length` about its `center`, a
`Disk`/`Annulus` by its radii), independent of any square-padding applied to the viewing
window for display purposes.

#### Scenario: Disk includes its boundary

- **WHEN** a point satisfies `|z - center| == radius` for a `Disk(radius, center)`
- **THEN** the point is classified as inside the domain

#### Scenario: Annulus includes both boundaries

- **WHEN** a point satisfies `|z - center| == inner_radius` or `|z - center| == outer_radius`
- **THEN** the point is classified as inside the annulus

#### Scenario: Rectangle includes its edges

- **WHEN** a point lies exactly on a rectangle edge (real or imaginary extent equals the half-dimension)
- **THEN** the point is classified as inside the rectangle

#### Scenario: Non-square rectangle membership ignores square padding

- **WHEN** a `Rectangle(re_length, im_length)` with `re_length != im_length` is constructed with the default square viewing window, and a point lies within the padded window but outside the actual `re_length x im_length` rectangle (e.g. `0 + 1.5j` for `Rectangle(4, 2)`)
- **THEN** the point is classified as OUTSIDE the rectangle
