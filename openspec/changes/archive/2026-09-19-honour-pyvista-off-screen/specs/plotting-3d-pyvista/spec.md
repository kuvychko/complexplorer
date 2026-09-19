## ADDED Requirements

### Requirement: A session-level off-screen instruction is honoured

PyVista exposes a session-wide off-screen switch — `pyvista.OFF_SCREEN`, which the
`PYVISTA_OFF_SCREEN` environment variable sets. Every PyVista renderer in this library SHALL
render off-screen when that switch is set, regardless of the per-call `interactive` default, so
that setting it once for a script, a session or a CI job is sufficient to prevent windows.

A renderer SHALL render off-screen when the call requests it (`interactive=False`, or a filename
is given) **or** the session-level switch is set. An explicit `interactive=False` SHALL continue
to render off-screen whether or not the switch is set.

#### Scenario: The global switch prevents a window

- **WHEN** `pyvista.OFF_SCREEN` is true and a PyVista renderer is called without passing
  `interactive`
- **THEN** the scene is rendered off-screen and no interactive window is opened

#### Scenario: The environment variable has the same effect

- **WHEN** `PYVISTA_OFF_SCREEN` is set in the environment, so PyVista's global switch is true, and
  a PyVista renderer is called without passing `interactive`
- **THEN** the scene is rendered off-screen

#### Scenario: An explicit request is still honoured

- **WHEN** a renderer is called with `interactive=False` while the session-level switch is unset
- **THEN** the scene is rendered off-screen, as before
