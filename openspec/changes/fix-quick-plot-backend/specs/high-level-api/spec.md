# High-Level API

## MODIFIED Requirements

### Requirement: Mode- and backend-dispatching plot

The library SHALL provide a `plot` function that selects among 2D, 3D landscape, and Riemann
renderings and between matplotlib and PyVista backends, defaulting the domain and colormap
when not supplied. For 3D and Riemann modes it SHALL default to the **PyVista** backend when
PyVista is installed (per the backend policy), using matplotlib only when PyVista is absent
or matplotlib is explicitly requested. The `backend` selector SHALL NOT be forwarded to the
underlying renderer.

#### Scenario: Mode selects the renderer

- **WHEN** `plot` is called with mode `2d`, `3d`, or `riemann`
- **THEN** the corresponding renderer is invoked and its native result (matplotlib axes or PyVista plotter) is returned

#### Scenario: PyVista is the default for 3D and Riemann when available

- **WHEN** a 3D or Riemann plot is requested and PyVista is installed, without an explicit
  backend
- **THEN** the PyVista renderer is used (and renderer-specific options such as a modulus
  scaling mode are accepted)

#### Scenario: matplotlib is used only as a fallback or when explicitly requested

- **WHEN** a 3D or Riemann plot is requested with PyVista absent, or with
  `backend="matplotlib"`
- **THEN** the (deprecated) matplotlib renderer is used and the `backend` selector is not
  forwarded to it

#### Scenario: Unknown mode is rejected

- **WHEN** `plot` is called with a mode other than `2d`, `3d`, or `riemann`
- **THEN** an error is raised naming the unknown mode
