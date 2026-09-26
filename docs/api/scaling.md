# Modulus scaling

How `|f(z)|` becomes height or displacement. The most consequential setting in any 3D render.

::: complexplorer.ModulusScaling
::: complexplorer.get_scaling_preset

## Normalization

Sea level sits at `|f| = 1` for every self-dual transfer, so the arbitrary constant in front of a
function changes the shape of a relief rather than only its labels. These compute the constant that
puts the area-weighted geometric mean of `|f|` over the sphere at 1, which removes that dependence.

`normalization_constant` is the exact closed form for a rational function, and needs a **complete**
divisor. `sampled_normalization_constant` estimates the same quantity from samples, needs no divisor,
and is what the ornament path uses by default.

::: complexplorer.normalization_constant
::: complexplorer.sampled_normalization_constant
