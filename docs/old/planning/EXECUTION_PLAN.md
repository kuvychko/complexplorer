# Comprehensive Execution Plan
## Complexplorer Technical Improvements + MkDocs Documentation

**Created**: 2025-10-14
**Status**: Ready for execution
**Estimated Total Time**: 60-80 hours

---

## Overview

This plan covers autonomous execution of:
1. Remaining Quick Wins (QW-6, QW-7, QW-8)
2. Phase 1: Code Quality Improvements (P1-1 through P1-6)
3. MkDocs Material setup with GitHub Pages
4. Documentation content creation and migration
5. Testing improvements (visual regression, property-based tests)

**Key Principle**: Interleave code improvements with documentation work to maintain momentum and provide immediate value.

---

## Execution Strategy

### Week 1: Critical Code Improvements + MkDocs Setup (20-25 hours)

#### Day 1-2: CRITICAL - Colormap Duplication (8 hours)
**Task**: P1-1 - Eliminate 800+ lines of colormap duplication

**Why First**: This is the single biggest code quality issue and blocks other improvements.

**Approach**:
1. Create `BasePhasePortrait` class with shared hsv() logic
2. Extract common parameters (v_base, v_contrast, r_linear_step, etc.)
3. Refactor Phase, OklabPhase, PerceptualPastel to inherit from base
4. Update all 7+ colormap classes systematically
5. Run pytest after each class migration
6. Update tests to verify behavior unchanged

**Deliverable**: Single source of truth for phase portrait colormaps

**Testing**: Run full test suite, visual inspection of examples

---

#### Day 2-3: MkDocs Infrastructure Setup (6 hours)

**Task**: Set up MkDocs Material with GitHub Pages deployment

**Steps**:

1. **Install Dependencies** (30 min)
   ```bash
   uv pip install mkdocs-material mkdocs-jupyter pymdown-extensions
   ```

2. **Create mkdocs.yml Configuration** (1 hour)
   ```yaml
   site_name: Complexplorer
   site_description: Beautiful visualizations of complex functions
   site_url: https://USERNAME.github.io/complexplorer/
   repo_url: https://github.com/USERNAME/complexplorer

   theme:
     name: material
     palette:
       - scheme: default
         primary: deep purple
         accent: amber
     features:
       - navigation.tabs
       - navigation.sections
       - navigation.expand
       - search.suggest
       - content.code.copy

   plugins:
     - search
     - mkdocs-jupyter

   markdown_extensions:
     - pymdownx.highlight
     - pymdownx.superfences
     - pymdownx.arithmatex:
         generic: true
     - admonition
     - toc:
         permalink: true

   nav:
     - Home: index.md
     - Getting Started:
       - Installation: getting-started/installation.md
       - Quick Start: getting-started/quickstart.md
     - User Guide:
       - Domains: user-guide/domains.md
       - Colormaps: user-guide/colormaps.md
       - 2D Plotting: user-guide/plotting-2d.md
       - 3D Plotting: user-guide/plotting-3d.md
       - Riemann Sphere: user-guide/riemann.md
     - Examples:
       - Gallery: examples/gallery.md
       - Notebooks: examples/notebooks.md
     - API Reference:
       - Core: api/core.md
       - Plotting: api/plotting.md
       - Export: api/export.md
     - Development:
       - Contributing: development/contributing.md
       - Architecture: development/architecture.md
   ```

3. **Create Directory Structure** (30 min)
   ```
   docs/
   ├── index.md
   ├── getting-started/
   │   ├── installation.md
   │   └── quickstart.md
   ├── user-guide/
   │   ├── domains.md
   │   ├── colormaps.md
   │   ├── plotting-2d.md
   │   ├── plotting-3d.md
   │   └── riemann.md
   ├── examples/
   │   ├── gallery.md
   │   └── notebooks/
   ├── api/
   │   ├── core.md
   │   ├── plotting.md
   │   └── export.md
   ├── development/
   │   ├── contributing.md
   │   └── architecture.md
   └── old/  # Backup of existing docs
   ```

4. **Move Existing Docs to Backup** (30 min)
   - Move docs/planning/ → docs/old/planning/
   - Move docs/review/ → docs/old/review/
   - Move docs/gallery/ → docs/old/gallery/
   - Move root *.md files (except README) to docs/old/

5. **Create GitHub Actions Workflow** (1 hour)
   `.github/workflows/docs.yml`:
   ```yaml
   name: Deploy Documentation

   on:
     push:
       branches:
         - main
     workflow_dispatch:

   permissions:
     contents: write

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: 3.11
         - run: pip install mkdocs-material mkdocs-jupyter pymdown-extensions
         - run: mkdocs gh-deploy --force
   ```

6. **Create Minimal Index Page** (30 min)
   - Simple landing page with project overview
   - Link to getting started
   - Link to GitHub repo

7. **Test Local Build** (30 min)
   ```bash
   mkdocs serve
   # Visit http://127.0.0.1:8000
   ```

**Deliverable**: Working MkDocs site with structure, deployable to GitHub Pages

---

#### Day 3-4: Quick Wins QW-6, QW-7, QW-8 (6 hours)

**QW-6: Add Return Type Hints** (2 hours)
- Add to 23+ functions missing return types
- Focus on public API functions first
- Run mypy to verify correctness

**QW-7: Standardize Exception Hierarchy** (1.5 hours)
- Create custom exceptions in `complexplorer/exceptions.py`
- Replace generic ValueError/TypeError with specific exceptions
- Document exception hierarchy in docstrings

**QW-8: Colormap Validation** (2 hours)
- Add parameter validation to all colormap __init__ methods
- Use pydantic validators or custom validation functions
- Ensure helpful error messages

**Testing**: Run pytest after each quick win

---

#### Day 4-5: Core Documentation Content (6 hours)

**Create Essential Pages**:

1. **getting-started/installation.md** (1 hour)
   - Installation with uv, pip, conda
   - Optional dependencies (PyVista, PyQt6)
   - Verification steps

2. **getting-started/quickstart.md** (2 hours)
   - 5-minute tutorial
   - Basic domain creation
   - Simple phase portrait
   - Interactive examples with code

3. **user-guide/domains.md** (1 hour)
   - Rectangle, Disk, Annulus, CompositeDomain
   - Code examples for each
   - Visual diagrams

4. **user-guide/colormaps.md** (2 hours)
   - Overview of available colormaps
   - When to use each type
   - Parameter explanations
   - Visual comparison gallery

**Deliverable**: Users can install and create first visualization

---

### Week 2: Phase 1 Completion + Extended Documentation (20-25 hours)

#### Day 6-7: Phase 1 Remaining Tasks (10 hours)

**P1-4: OkLCh Gamut Clipping** (30 min)
- Implement proper gamut clipping in OklabPhase
- Test with edge cases

**P1-3: Consolidate Plot Validation** (3 hours)
- Create shared validation module
- Extract common parameter checks
- Reduce duplication across plot functions

**P1-6: Standardize Mask Application** (2 hours)
- Unified mask handling across all colormaps
- Consistent out-of-domain color application

**P1-5: Add Logging Framework** (4 hours)
- Set up Python logging with named loggers
- Add debug/info/warning levels throughout
- Document logging configuration for users

**P1-2: Standardize Parameter Naming** (BREAKING, 3 hours)
- n_phi → phase_sectors
- n/N → resolution
- Update all code, tests, examples
- Document breaking changes

**Testing**: Full pytest suite after all changes

---

#### Day 8-9: User Guide Completion (8 hours)

**user-guide/plotting-2d.md** (2 hours)
- plot() function
- pair_plot() for domain/codomain comparison
- Customization options
- Examples with different colormaps

**user-guide/plotting-3d.md** (2 hours)
- Matplotlib 3D landscapes
- PyVista high-performance rendering
- Material and lighting parameters
- Performance tips

**user-guide/riemann.md** (2 hours)
- Stereographic projection explanation
- riemann() and riemann_pv() functions
- Modulus scaling options
- Visual examples

**examples/gallery.md** (2 hours)
- Curated gallery of beautiful visualizations
- Code snippets for each
- Links to full notebooks
- Organized by complexity

---

#### Day 10: Testing Infrastructure (4 hours)

**T-1: Visual Regression Tests** (2 hours)
- Install pytest-mpl
- Create baseline images for key plots
- Add visual comparison tests
- Document how to update baselines

**T-2: Property-Based Tests** (2 hours)
- Install hypothesis
- Add property tests for:
  - Domain.contains() properties
  - Colormap output bounds (HSV in [0,1])
  - Stereographic projection roundtrips

---

### Week 3: API Reference + Polish (15-20 hours)

#### Day 11-12: API Documentation (8 hours)

**Strategy**: Extract docstrings into organized API reference

**api/core.md** (3 hours)
- Domain classes (Rectangle, Disk, Annulus, etc.)
- Colormap base class and all implementations
- ModulusScaling and presets
- Helper functions

**api/plotting.md** (3 hours)
- All plot functions with full signatures
- Parameter descriptions
- Return types
- Examples

**api/export.md** (2 hours)
- STL export functions
- Mesh utilities
- 3D printing parameters

**Tool**: Use mkdocstrings plugin for automatic docstring extraction

---

#### Day 13-14: Examples and Notebooks (8 hours)

**Migrate Example Notebooks** (4 hours)
- Convert examples/ notebooks to docs/examples/notebooks/
- Add markdown explanations
- Test with mkdocs-jupyter plugin
- Create index with descriptions

**Create Tutorial Notebooks** (4 hours)
- Beginner: Simple phase portrait
- Intermediate: Custom colormap
- Advanced: Riemann sphere with STL export
- Test all notebooks execute correctly

---

#### Day 15: Final Polish (4 hours)

**development/contributing.md** (1 hour)
- Development setup
- Testing guidelines
- Code style
- PR process

**development/architecture.md** (1 hour)
- Package structure explanation
- Design decisions
- Import layer diagram

**Final Review** (2 hours)
- Check all internal links work
- Verify code examples run
- Test GitHub Pages deployment
- Fix any broken references

---

## Testing Strategy Throughout

### After Each Code Change:
```bash
# Run core tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/unit/ --cov=complexplorer --cov-report=term-missing

# Visual inspection
python examples/basic_examples.py
```

### After Documentation Changes:
```bash
# Local preview
mkdocs serve

# Check for broken links
mkdocs build --strict
```

### Before Commits:
- All tests passing
- No regression in visual output
- Documentation builds without warnings

---

## Commit Strategy

### Small, Focused Commits:
1. **After P1-1 completion**: "Eliminate colormap duplication with BasePhasePortrait"
2. **After MkDocs setup**: "Set up MkDocs Material with GitHub Pages deployment"
3. **After QW-6,7,8**: "Add return type hints, exception hierarchy, and colormap validation"
4. **After P1 completion**: "Complete Phase 1 code quality improvements"
5. **After core docs**: "Add getting started and user guide documentation"
6. **After API docs**: "Add comprehensive API reference"
7. **After examples**: "Migrate and enhance example notebooks"

### Commit Message Format:
```
<type>: <short description>

<detailed explanation>

- Bullet points of key changes
- Reference to issue numbers if applicable

Breaking changes: <if any>
```

---

## Risk Mitigation

### Code Changes:
- **Risk**: Breaking existing user code
- **Mitigation**: Document all breaking changes in MIGRATION_GUIDE_V2.md

### Documentation:
- **Risk**: Documentation diverges from code
- **Mitigation**: Test all code examples, use automated docstring extraction

### Testing:
- **Risk**: Visual regression not caught
- **Mitigation**: Manual visual inspection + pytest-mpl baselines

---

## Success Criteria

### Code Quality:
- [ ] No code duplication in colormaps (P1-1)
- [ ] Consistent parameter naming (P1-2)
- [ ] All functions have return type hints (QW-6)
- [ ] Custom exception hierarchy (QW-7)
- [ ] Validated colormap parameters (QW-8)
- [ ] Logging framework in place (P1-5)
- [ ] All tests passing (135+ tests)

### Documentation:
- [ ] MkDocs site deployed to GitHub Pages
- [ ] Complete getting started guide
- [ ] Comprehensive user guide
- [ ] Full API reference
- [ ] Working example notebooks
- [ ] No broken links
- [ ] Mobile-friendly theme

### Testing:
- [ ] Visual regression test framework
- [ ] Property-based tests for core functions
- [ ] All examples execute without errors

---

## Questions Before Starting

None - plan is comprehensive and autonomous. Will proceed with execution and report progress.

---

## Notes

- All existing docs backed up to docs/old/
- Breaking changes documented in MIGRATION_GUIDE_V2.md
- User will review and approve commits
- Visual regression via manual checks initially, automated tests added during execution
