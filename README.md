# Elasto-plasticity_modules_dealii
A modular class containing elasto-plastic material models (Hill-Plasticity) with straightforward extension to various hardening laws.

## What it does
We offer a framework to capture elasto-plastic material models up to anisotropic Hill-plasticity with various hardening laws. The framework can be found in the exemplary `MaterialModel.h` file and also contains subiterations on the qp level. Different hardenig laws defined by the hardening stress `R` and an evolution equation for the internal hardening variable `alpha` can be defined by only three equations (R, alpha, d_R_d_gamma in `elpl_equation_list.h`). The algorithm is general enough to produce quadratic convergence for any such defined hardening law.

@todo Hill-plasti currently only superlinear on qp level
@todo note on low efficiency, recommended for testing
