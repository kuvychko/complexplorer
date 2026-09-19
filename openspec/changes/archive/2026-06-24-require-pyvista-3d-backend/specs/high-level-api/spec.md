# High-Level API

## MODIFIED Requirements

### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `plot` function that selects among 2D, 3D landscape, and Riemann
renderings, defaulting the domain and colormap when not supplied. 2D uses matplotlib; **3D and
Riemann use PyVista, which is a required dependency** — there is no matplotlib 3D backend. The
`backend` selector SHALL NOT be forwarded to the underlying renderer, and requesting
`backend="matplotlib"` for a 3D or Riemann mode SHALL raise a clear error stating the
matplotlib 3D backend was removed.

#### Scenario: Mode selects the renderer

- **WHEN** `plot` is called with mode `2d`, `3d`, or `riemann`
- **THEN** the corresponding renderer is invoked (matplotlib for 2D, PyVista for 3D/Riemann) and its native result (matplotlib axes or PyVista plotter) is returned

#### Scenario: 3D and Riemann always use PyVista

- **WHEN** a 3D or Riemann plot is requested without an explicit backend
- **THEN** the PyVista renderer is used (and renderer-specific options such as a modulus scaling mode are accepted)

#### Scenario: Requesting the removed matplotlib 3D backend errors

- **WHEN** `plot` is called with mode `3d` or `riemann` and `backend="matplotlib"`
- **THEN** an error is raised explaining the matplotlib 3D backend was removed in 3.0

#### Scenario: Unknown mode is rejected

- **WHEN** `plot` is called with a mode other than `2d`, `3d`, or `riemann`
- **THEN** an error is raised naming the unknown mode
