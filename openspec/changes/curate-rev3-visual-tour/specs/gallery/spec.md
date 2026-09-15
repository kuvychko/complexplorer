## MODIFIED Requirements

### Requirement: Images are reproducible best-effort

Rendered `portrait.png` files SHALL be written with image metadata stripped (no software or
timestamp tags) so they are reproducible within an environment. Pixel bytes are NOT guaranteed
across environments or rendering-library versions; the byte-stable guarantee applies to the
manifest, not the images. Portraits SHALL be framed so that every axis label and tick label they
draw lies inside the image.

#### Scenario: Portrait metadata carries no timestamp

- **WHEN** a portrait PNG is written
- **THEN** it contains no embedded creation timestamp or software-version tag

#### Scenario: Axis labels are not clipped

- **WHEN** a portrait is written for a domain whose axes are labelled
- **THEN** the image contains the whole label (the `Im(z)` label is not cut off at the left edge) and carries no excessive empty margin
