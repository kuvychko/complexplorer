## MODIFIED Requirements

### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `quick_plot` function — the single high-level exploration entry
point — that selects among 2D, 3D landscape, and Riemann renderings, defaulting the domain
(`Rectangle(4, 4)`) and colormap (enhanced `Phase`) when not supplied. 2D uses matplotlib;
**3D and Riemann use PyVista, which is a required dependency** — there is no matplotlib 3D
backend. The `backend` selector SHALL NOT be forwarded to the underlying renderer, and
requesting `backend="matplotlib"` for a 3D or Riemann mode SHALL raise a clear error stating
the matplotlib 3D backend was removed. A caller-supplied `domain` SHALL be forwarded to the
selected renderer in every mode, including Riemann. No alias names for `quick_plot` are exported.

#### Scenario: Quick plot with only a function

- **WHEN** `quick_plot` is called with only a function
- **THEN** a `Rectangle(4, 4)` domain and an enhanced `Phase` colormap are applied and a 2D plot is produced

#### Scenario: Mode selects the renderer

- **WHEN** `quick_plot` is called with mode `2d`, `3d`, or `riemann`
- **THEN** the corresponding renderer is invoked (matplotlib for 2D, PyVista for 3D/Riemann) and its native result (matplotlib axes or PyVista plotter) is returned

#### Scenario: 3D and Riemann always use PyVista

- **WHEN** a 3D or Riemann plot is requested without an explicit backend
- **THEN** the PyVista renderer is used (and renderer-specific options such as a modulus scaling mode are accepted)

#### Scenario: Requesting the removed matplotlib 3D backend errors

- **WHEN** `quick_plot` is called with mode `3d` or `riemann` and `backend="matplotlib"`
- **THEN** an error is raised explaining the matplotlib 3D backend was removed in 3.0

#### Scenario: Unknown mode is rejected

- **WHEN** `quick_plot` is called with a mode other than `2d`, `3d`, or `riemann`
- **THEN** an error is raised naming the unknown mode

#### Scenario: A supplied domain is forwarded in Riemann mode

- **WHEN** `quick_plot` is called with mode `riemann` and an explicit `domain`
- **THEN** that domain is passed through to `riemann_pv` (used to mask the sphere) rather than being discarded in favor of the default
