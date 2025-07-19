# Complexplorer Documentation Improvement Plan

## Current State Analysis

### Existing Documentation Structure
1. **README.md** - Main documentation with:
   - Project overview and philosophy
   - Library structure overview
   - Installation instructions
   - Links to example notebooks
   - Embedded gallery (static images)
   - Future work section

2. **Example Notebooks**:
   - `plots_example.ipynb` - Basic functionality overview
   - `domains_cmaps_example.ipynb` - Detailed domain and color map examples
   - `pyvista_demo.ipynb` - PyVista 3D visualization (new, not yet integrated)

3. **Gallery** - Static PNG images in `examples/gallery/` folder

### Current Issues
- Gallery is embedded directly in README making it long and harder to navigate
- No clear separation between API reference and tutorials
- No interactive examples or live demos
- Limited explanation of mathematical concepts
- No contribution guidelines
- Missing examples for advanced use cases

## Proposed Improvements

### 1. Documentation Structure Reorganization

Create a proper documentation structure:
```
docs/
├── index.md                    # Landing page with quick start
├── installation.md             # Detailed installation guide
├── tutorials/
│   ├── getting_started.md      # Basic tutorial
│   ├── domains_guide.md        # Working with domains
│   ├── colormaps_guide.md      # Understanding color maps
│   ├── plotting_guide.md       # 2D and 3D plotting
│   └── advanced_topics.md      # Complex compositions, custom domains
├── gallery/
│   ├── index.md               # Gallery landing page
│   ├── basic_functions.md     # Elementary complex functions
│   ├── phase_portraits.md     # Different phase portrait styles
│   ├── 3d_visualizations.md   # Analytic landscapes & Riemann sphere
│   └── custom_examples.md     # User-contributed examples
├── api/
│   ├── domains.md             # Domain classes reference
│   ├── colormaps.md           # Color map classes reference
│   ├── plotting_2d.md         # 2D plotting functions
│   ├── plotting_3d.md         # 3D plotting functions
│   └── utilities.md           # Helper functions
├── mathematical_background.md  # Theory behind visualizations
└── contributing.md            # How to contribute

complexplorer/
├── __init__.py
├── domain.py
├── cmap.py
├── plots_2d.py
├── plots_3d.py
└── funcs.py
```

### 2. Enhanced Gallery Implementation

#### Option A: Static Gallery with Better Organization
- Create a dedicated gallery section separate from README
- Organize examples by category
- Add code snippets alongside each image
- Include mathematical description of each function

#### Option B: Interactive Gallery (Recommended)
- Use Jupyter Book or Sphinx-Gallery for documentation
- Convert notebooks to gallery examples
- Auto-generate gallery from Python scripts
- Include interactive plots with Plotly/Bokeh backend

#### Option C: Hybrid Approach
- Static images for README quick preview
- Full interactive gallery in documentation
- GitHub Pages hosting for web-based gallery

### 3. Gallery Content Organization

```
Gallery/
├── Basic Functions
│   ├── Identity: f(z) = z
│   ├── Powers: f(z) = z^n
│   ├── Exponential: f(z) = e^z
│   ├── Logarithm: f(z) = log(z)
│   └── Trigonometric: sin(z), cos(z), tan(z)
├── Rational Functions
│   ├── Simple poles: f(z) = 1/z
│   ├── Multiple poles: f(z) = 1/z^n
│   ├── Rational examples: f(z) = (z-1)/(z^2+z+1)
│   └── Möbius transformations
├── Special Functions
│   ├── Gamma function
│   ├── Riemann zeta
│   ├── Elliptic functions
│   └── Modular forms
├── Visualization Techniques
│   ├── Phase portraits (all variants)
│   ├── Chessboard patterns
│   ├── Logarithmic rings
│   └── Custom color maps
├── 3D Visualizations
│   ├── Analytic landscapes
│   ├── Riemann sphere projections
│   ├── Domain/codomain comparisons
│   └── PyVista renderings
└── Advanced Examples
    ├── Domain compositions
    ├── Conformal mappings
    ├── Branch cuts visualization
    └── Essential singularities
```

### 4. Implementation Plan

#### Phase 1: Documentation Infrastructure (Week 1)
- [ ] Set up documentation framework (Sphinx/MkDocs)
- [ ] Create directory structure
- [ ] Configure GitHub Pages deployment
- [ ] Set up auto-generation from docstrings

#### Phase 2: Content Migration (Week 2)
- [ ] Split README into separate documentation pages
- [ ] Convert notebooks to documentation format
- [ ] Extract and organize existing gallery images
- [ ] Write missing documentation sections

#### Phase 3: Gallery Enhancement (Week 3-4)
- [ ] Create gallery generator script
- [ ] Add code examples for each gallery item
- [ ] Generate additional examples for missing functions
- [ ] Add mathematical descriptions

#### Phase 4: Interactive Features (Week 5-6)
- [ ] Integrate PyVista examples
- [ ] Add interactive widgets (if using Jupyter Book)
- [ ] Create downloadable example scripts
- [ ] Add "try it yourself" sections

### 5. Gallery Generator Script Concept

```python
# gallery_generator.py
import complexplorer as cp
import numpy as np
from pathlib import Path

class GalleryExample:
    def __init__(self, name, category, function, description, math_formula):
        self.name = name
        self.category = category
        self.function = function
        self.description = description
        self.math_formula = math_formula
    
    def generate_images(self, output_dir):
        """Generate all visualization types for this example"""
        # 2D phase portrait
        # 3D landscape
        # Riemann sphere
        # Save with consistent naming
        pass
    
    def generate_markdown(self):
        """Generate markdown documentation for this example"""
        pass

# Define all examples
examples = [
    GalleryExample(
        name="Identity Function",
        category="Basic Functions",
        function=lambda z: z,
        description="The identity function maps each point to itself",
        math_formula="f(z) = z"
    ),
    # ... more examples
]

# Generate gallery
for example in examples:
    example.generate_images("docs/gallery/images")
    example.generate_markdown()
```

### 6. README.md Restructuring

The new README should be concise and include:
- Brief description (keep Penrose quote)
- Key features
- Quick installation
- Simple example
- Link to full documentation
- Link to gallery preview (3-4 images max)
- Contributing guidelines summary

### 7. Additional Recommendations

1. **Add Binder integration** for running examples online
2. **Create a showcase section** for user-contributed visualizations
3. **Add performance tips** for large-scale visualizations
4. **Include troubleshooting guide**
5. **Add citation information** for academic use
6. **Create video tutorials** for complex topics
7. **Add comparison with similar libraries**

### 8. Documentation Hosting Options

1. **GitHub Pages** (Recommended)
   - Free hosting
   - Automatic deployment from main branch
   - Custom domain support
   - Good for static sites

2. **Read the Docs**
   - Automatic building
   - Version support
   - Search functionality
   - Good for Sphinx docs

3. **GitBook**
   - Modern interface
   - Good for tutorials
   - Limited customization

### 9. Success Metrics

- Reduced README length by 70%
- Increased example coverage to 50+ functions
- Documentation build time < 5 minutes
- Mobile-friendly gallery
- Search functionality
- Clear navigation structure

## Next Steps

1. **Immediate** (This PR):
   - Create basic gallery structure
   - Move existing gallery images
   - Update README with links

2. **Short term** (Next PR):
   - Set up documentation framework
   - Migrate content
   - Create gallery generator

3. **Long term**:
   - Add interactive features
   - Expand example coverage
   - Community contributions