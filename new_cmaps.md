Below are palette families you can add as plug-ins. Each keeps the core mapping
$r=|f(z)|,\ \theta=\arg f(z),\ \rho=\log_b r$ (pick $b=e$ or 10), and a smooth “sawtooth” $T(\rho)=\mathrm{frac}(\rho)$ (optionally softened with a raised-cosine).

---

# 1) Perceptual pastel (OkLCh “ring”)

* **Meaning:** hue = phase; lightness encodes modulus (monotone); thin iso-|f| bands.
* **Mapping (OkLCh → sRGB):**
  `H = θ (in degrees)`
  `L = 0.55 + 0.25 * (T(ρ) - 0.5)`  (gentle bands)
  `C = 0.10–0.20` (low chroma pastels)
* **Look:** elegant, non-fluorescent, print-friendly.
* **Notes:** Using OkLCh (or CIE LCh) keeps perceived lightness uniform as hue cycles, avoiding rainbow “heatmap” glare.

# 2) Analogous wedge (compressed hue range)

* **Meaning:** hue still tracks phase, but only over a wedge of the color wheel.
* **Mapping (HSL/HSV):**
  `H = H0 + w * θ/(2π)` with `w ∈ [0.2, 0.5]`
  `S = 0.25–0.45` (fixed, muted)
  `V = 0.55 + 0.35 * g(ρ)` with `g` monotone (e.g., sigmoid(log r))
* **Look:** “Ocean” (teal→navy), “Aubergine” (violet→maroon), “Amber-slate.”
* **Why:** keeping hues analogous kills the tie-dye vibe but preserves winding (phase).

# 3) Diverging warm–cool (phase-signed)

* **Meaning:** positive phases lean warm, negative phases cool; |f| in lightness.
* **Mapping (OkLCh):** pick anchors `H_warm≈30°` (amber), `H_cool≈220°` (indigo).
  `a = sin θ` in \[−1,1]; interpolate hue `H = mix(H_cool, H_warm, (a+1)/2)`
  `L = 0.5 + 0.3 * h(ρ)`; `C = 0.06–0.16`
* **Look:** refined, cartographic; real/imag axes become naturally emphasized.

# 4) Isoluminant hue + contour lines

* **Meaning:** hue = phase only; modulus shown via subtle topographic lines (no value ramp).
* **Mapping:**
  Base: `L = 0.6` constant, `C = 0.12–0.18`, `H = θ`.
  Overlay: draw thin lines where `T(ρ) ≈ 0` using `(1−exp(−(T/σ)^2))` as opacity.
* **Look:** calm “silk” backgrounds with clean bands; extremely readable zeros/poles.

# 5) Cubehelix-phase

* **Meaning:** encodes θ into a cubehelix angle; |f| in lightness; cubehelix is linear-light and CMYK-safe.
* **Mapping:**
  `L = 0.5 + 0.35 * g(ρ)`; `φ = θ + φ0`
  `r,g,b = cubehelix(L, φ, a=−0.5…0.5)` (standard formula).
* **Look:** scientific, subdued gradients; great for print and grayscale conversions.

# 6) Ink & paper (nearly monochrome with tints)

* **Meaning:** modulus in lightness; phase as a very small chroma tint.
* **Mapping (OkLCh):**
  `L = 0.35 + 0.5 * g(ρ)`
  `H = θ`, `C = 0.02–0.06`
  Optional: phase stripes via `L += ε * cos(k θ)` (k=6–12, ε≈0.03).
* **Look:** classy poster/etching feel; almost grayscale, just enough color to read phase.

# 7) Earth-tone topographic

* **Meaning:** modulus as terrain; phase as soil/water tint.
* **Mapping:**
  `L = 0.4 + 0.4 * g(ρ)` with gentle hillshade `+ 0.07 * cos(2π T(ρ))`
  `H = mix(200°, 30°, (sin θ + 1)/2)`
  `C = 0.05–0.12`
* **Look:** maps/landforms aesthetic; zeros/poles pop like summits/sinks.

# 8) Four-quadrant smooth blend

* **Meaning:** map the four principal arguments (0, π/2, π, 3π/2) to four tasteful anchors; interpolate on the circle; modulus in lightness.
* **Mapping:** anchors (OkLCh):
  `H = {10°, 120°, 210°, 300°}` with `C≈0.10`, `L` from `g(ρ)`
  Interpolate across θ with circular spline to avoid seams.
* **Look:** geometric, Bauhaus-ish, reduced palette.

---

## Implementation tips (drop-in)

* Define once:

  ```python
  def soft_sawtooth(x, sharp=0.6):
      # x in R, returns [0,1) with softened corners
      t = x - np.floor(x)
      return 0.5 - 0.5*np.cos(2*np.pi*(t**sharp))
  ```
* Perceptual spaces: do the math in **OkLCh** (or CIE LChab) and convert to sRGB at the end. Keep `C` small and clip safely.
* Gentle bands: avoid hard `frac`; modulate with raised cosine:
  `band = 0.5*(1 - cos(2π * T(ρ)))`, then use `L = L0 + a*(band - 0.5)`.
* **Restrict gamut** for print: keep `0.45 ≤ L ≤ 0.75`, `C ≤ 0.18` (OkLCh). Offer a “CMYK-safe” toggle that desaturates and slightly lifts mid-tones.
* **Adaptive chroma:** `C = C0 / (1 + r^α)` desaturates far from the interesting action; or `C = C0 * sigmoid(−log r)` to emphasize neighborhoods of zeros/poles.
* **Derivative shading (optional):** add `+ β * norm_log(|f'(z)|)` to lightness to show local distortion without neon colors.

---

## Suggested presets (names + knobs)

* **Silk Ring (pastel):** OkLCh; `C=0.14`, `L0=0.6`, band `a=0.12`.
* **Ocean Wedge:** HSL; `H0=200°`, `w=0.35`, `S=0.35`, `V(sigmoid)`.
* **Amber–Indigo Diverge:** OkLCh; warm 30°, cool 220°, `C=0.12`, `L(sigmoid)`.
* **Topo Ink:** nearly monochrome; OkLCh `C=0.04`, bands on `L`.
* **Cubehelix Phase:** standard cubehelix with `φ0=−π/6`, `a=−0.4`.
