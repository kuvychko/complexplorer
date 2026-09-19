## Why

3.0 is the release that fixes the public surface, and two parts of it are still provisional.

**The typing contract is wrong where it is most visible.** `quick_plot` is annotated
`Callable[[complex], complex]`, but every renderer in the library calls the user's function with a
numpy array. A user who types their vectorized function correctly is told it does not fit; a user
who believes the annotation writes a scalar function that is silently broadcast. The package ships
`py.typed`, so this annotation is a promise to every type checker downstream. Three public
functions have no return annotation at all, and `Presets`' three methods return untyped dicts.

**The validation surface is inconsistent, and in one place it is wrong.** The domain classes
validate properly — `Rectangle(-4, 4)`, `Disk(radius=-1)` and `Annulus(3, 1)` each raise a
`ValidationError` that names the problem. The colormaps do not:

- `cp.Phase(phase_sectors=0)` raises **`ZeroDivisionError`** from deep inside a render, not at
  construction. CLAUDE.md requires the library's own exception types rather than bare Python ones,
  and this is the case that most looks like a user typo.
- `cp.Phase(phase_sectors=-1)` and `cp.Phase(phase_sectors=2.5)` are accepted silently and produce
  a meaningless sector count.

**Naming.** `cp.Presets` (render configuration) and `cp.catalog` (functions) are easy to confuse,
and the closeout asks for this to be settled before release. Published 2.0.0 has no `Presets` at
all — it used `publication_preset()` — so renaming it now breaks nobody who installed from PyPI.
After 3.0 ships it would be a breaking change for real users.

## What Changes

**Naming**
- Rename `cp.Presets` to `cp.PlotPresets`, and teach the distinction in one sentence wherever both
  appear: *`PlotPresets` configures a render; `catalog` supplies a function.* The documentation
  site published in C3 references `Presets` on several pages and is updated with the rename.

**The typing contract**
- Add a `ComplexFunction` protocol describing what the library actually calls: an array in, an
  array out, with scalars accepted. Replace `Callable[[complex], complex]` wherever it appears.
- Add `TypedDict`s for `domain_spec`, `cmap_spec` and `scaling_spec`, and for the singularity
  records, so the catalog's interchange format is checkable rather than `dict`.
- Give every public callable an explicit return type, including `quick_plot`, the `PlotPresets`
  methods, the `cp.ee` plotters and the gallery entry points.
- Document `TransferFunction.__call__`'s scalar-versus-array behaviour, which is currently only
  discoverable by experiment.

**A type gate that is worth having**
- Run pyright over the **public surface** and over a small downstream fixture that imports
  complexplorer the way a user would, and gate it in CI.
- **Not** over the whole package: pyright in basic mode reports 86 errors there, and the three most
  suspicious were each checked and found to be false positives — two are narrowing limitations
  (a compound `if domain is None and z is None` guard, and a variable set and read under the same
  `verbose` condition), and the third is pyvista stub imprecision for a `tolerance` argument that
  genuinely exists. A gate dominated by numpy scalar-versus-array unions would be a tax that
  teaches people to add ignores. The gate covers what users can actually see.

**Validation and error messages**
- `phase_sectors` accepts positive integers, and anything else raises `ValidationError` naming the
  value received and what is accepted. `ZeroDivisionError` stops being reachable through the public
  API.
- Audit the `ValidationError` messages across the public surface so each names the offending value,
  the accepted values, and — where it replaces a 2.x name — the replacement.

**Discoverability**
- An API map page on the documentation site: one entry point per workflow, each with what it
  returns (`Axes`, `Figure`, `Plotter`, a mesh, a path, or `None`).
- A test that every name in `__all__` has a docstring. C3 already tests that every name reaches the
  API reference; this closes the other half.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `high-level-api`: `Presets` becomes `PlotPresets`; `quick_plot` and the preset methods gain
  accurate types and return contracts.
- `colormaps`: `phase_sectors` is validated at construction and raises `ValidationError` rather
  than failing later with `ZeroDivisionError`.
- `exceptions`: the public API raises only `ComplexplorerError` subclasses, and messages name the
  value and the accepted values.
- `function-presets`: the spec dictionaries and singularity records have declared shapes.
- `transfer-functions`: `__call__`'s scalar-versus-array behaviour is specified.
- `packaging`: CI gates the public typing surface with pyright.
- `docs`: the site carries an API map, and reflects the rename.

## Impact

- **Breaking, deliberately:** `cp.Presets` → `cp.PlotPresets`. Recorded in the migration guide by
  C6. Published 2.0.0 never exposed `Presets`, so no PyPI user is affected.
- **Code:** `complexplorer/api.py`, `core/colormap.py`, `core/presets.py`, `ee/transfer_function.py`,
  `gallery.py`, and a new typing module for the protocol and the `TypedDict`s.
- **CI:** a pyright job over the public surface and the downstream fixture.
- **Docs:** the rename across the pages C3 published, plus the new API map page.
- **Not here:** the migration guide's content and the README (C6).
