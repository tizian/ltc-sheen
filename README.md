<img src="https://github.com/tizian/ltc-sheen/raw/master/images/teaser.jpg" alt="Teaser">

# Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines

Source code of the talk "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines" by [Tizian Zeltner](https://tizianzeltner.com/), [Brent Burley](https://www.linkedin.com/in/brent-burley-56972557), and [Matt Jen-Yuan Chiang](https://mattchiangvfx.com) presented at SIGGRAPH 2022.

* The [`pbrt-v3`](pbrt-v3) directory contains our reference implementation of the final [sheen BRDF](pbrt-v3/src/materials/sheenltc.cpp) together with additional baselines and prior work that we compare against in the [supplementary material](https://tizianzeltner.com/projects/Zeltner2022Practical/supplemental.pdf).

* All code necessary to reproduce the LTC fitting is located in the [`fitting`](fitting) directory.

---

We introduce a new volumetric sheen BRDF that approximates scattering observed in surfaces covered with normally-oriented fibers. The implementation is based on a linearly transformed cosine (LTC) lobe [[Heitz et al. 2016]](https://eheitzresearch.wordpress.com/415-2/). To be precise, our cosine-weighted BRDF is

$$
  f_r(\omega_i, \omega_o) \cos\theta_o = C_\text{sheen} \cdot R_i \cdot D_o
  \left( \frac{\textbf{M}^{-1}_i \cdot \omega_o}{|| \textbf{M}^{-1}_i \cdot \omega_o ||} \right)
  \frac{| \textbf{M}^{-1}_i |}{|| \textbf{M}^{-1}_i \cdot \omega_o ||^3},
$$

where $D_o$ <span></span>is a normalized clamped cosine distribution and $C_\text{sheen}$ <span></span>is an artist-specified RGB scale. The directional hemispherical reflectance $R_i \in \mathbb{R}^{+}$ <span></span>and the linear transform $\textbf{M}^{-1}_{i} \in \mathbb{R}^{3\text{x}3}$ <span></span>both depend on the incident elevation angle $\theta_i$ <span></span>and a roughness parameter $\alpha$.

Their values are fit to a reference BRDF for a regular grid of 32 inputs for both $\alpha$ <span></span>and $\cos\theta_i$. For a good approximation, we found it sufficient to only fit $R_i$ <span></span>and two values $A_i, B_i$ <span></span>of the full matrix

$$
  \textbf{M}^{-1}_i = \begin{bmatrix}
    A_i & 0 & B_i \\
    0 & A_i & 0 \\
    0 & 0 & 1
  \end{bmatrix}.
$$

---

In terms of reference BRDFs we use two versions in this codebase (which are both impractical to use directly in a renderer):

1. The volumetric layer with a fiber-like SGGX phase function discussed in the talk abstract, where the cross section parameter $\sigma \text{ }\in\text{ } [0, 1]$ <span></span>is mapped to the roughness as $\alpha = \sqrt{\sigma}$. Importantly, this includes multiple-scattering which requires a costly stochastic evaluation via random walks. Further, we are only interested in its _reflected_ component which is poorly importance sampled in some configurations and thus can have unacceptably high variance. ([See code](https://github.com/tizian/ltc-sheen/blob/master/fitting/src/bsdfs/sheen_volume.h))

2. An analytic approximation of the above that is very cheap to evaluate. However it cannot directly be importance sampled which also leads to variance in practice. ([See code](https://github.com/tizian/ltc-sheen/blob/master/fitting/src/bsdfs/sheen_approx.h))

By fitting suitable LTC parameters we can sidestep all of these issues and arrive at BRDFs with efficient evaluation and perfect cosine-weighted importance sampling. Because the cosine is baked in (for sampling efficiency), the resulting BRDF is only approximately reciprocal, which we did not find problematic in practice.

We provide precomputed coefficient tables for both versions. They are also visualized below.

<br>

| Fitted BRDF | NumPy array | C++ array |
| --- | --- | --- |
| SGGX volume   | [`ltc_table_sheen_volume.npy`](https://github.com/tizian/ltc-sheen/blob/master/fitting/python/data/ltc_table_sheen_volume.npy) | [`ltc_table_sheen_volume.cpp`](https://github.com/tizian/ltc-sheen/blob/master/fitting/python/data/ltc_table_sheen_volume.cpp) |
| Approximation | [`ltc_table_sheen_approx.npy`](https://github.com/tizian/ltc-sheen/blob/master/fitting/python/data/ltc_table_sheen_approx.npy) | [`ltc_table_sheen_approx.cpp`](https://github.com/tizian/ltc-sheen/blob/master/fitting/python/data/ltc_table_sheen_approx.cpp) |

<br>

<p align="center">
  <img src="https://github.com/tizian/ltc-sheen/raw/master/images/coeffs_volume.jpg" alt="LTC coefficients (volume)">
  <img src="https://github.com/tizian/ltc-sheen/raw/master/images/coeffs_approx.jpg" alt="LTC coefficients (analytic approximation)">
</p>

Note how the left-most parts of the tables for version (1) are not smooth. The stochastically evaluated BRDF is particularly noisy in those $\theta_i, \alpha$ <span></span> configurations which leads to an unstable fitting process. However this does not cause any issues in practice as the reflectance $R_i$ <span></span> is also very small in those cases.

Version (2) produces lookup tables that are smooth in all regions, but at the cost of slightly higher approximation error compared to the original SGGX volume layer reference.
