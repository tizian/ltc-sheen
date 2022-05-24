<p align="center">
  <img src="https://github.com/tizian/ltc-sheen/raw/master/images/teaser2.jpg" alt="Teaser" style="width: 90%;">
</p>
  
# Reference pbrt-v3 implementation

We provide reference implementations for our proposed LTC sheen BRDF, together with baselines and prior work.

## Compilation

1. Clone the [`pbrt-v3`](https://github.com/mmp/pbrt-v3) repository:
  ```
  git clone --recursive https://github.com/mmp/pbrt-v3/
  ```
  
2. Copy over the contents of `src` from this repository into the respective cloned directory. Note that the file `src/core/api.cpp` needs to be overwritten.

3. Compile `pbrt-v3` normally, e.g. on Unix:
  ```
  mkdir build
  cd build
  cmake ..
  make -j
  ```

See the original [README](https://github.com/mmp/pbrt-v3/blob/master/README.md) for details.

## List of implemented BRDFs

### LTC Sheen

[`sheenltc.h`](src/materials/sheenltc.h) / [`sheenltc.cpp`](src/materials/sheenltc.cpp)
Our LTC fits based on either the volumetric SGGX or analytic approximation as references.

Scene format example:
```
Material "sheenltc" "color Csheen" [1 1 1] "float alpha" [0.5] "string type" [<type>]
```
where `<type>` should be replaced with either `"volume"` or `"approx"` to select which version should be used.

### Volumetric models

[`sheenvolume.h`](src/materials/sheenvolume.h) / [`sheenvolume.cpp`](src/materials/sheenvolume.cpp)
The fiber-like SGGX volumetric model we used as inspiration/reference of our LTC fits.
  
Scene format example:
```
Material "sheenvolume" "rgb albedo" [1 1 1] "float density" [1.0] "float sigma" [0.25] "integer maxBounces" [16]
```
  
Setting the `maxBounces` parameter to `1` also gives the interesting single-scattering only case.

### Prior work

[`sheenbaselines.h`](src/materials/sheenbaselines.h) / [`sheenbaselines.cpp`](src/materials/sheenbaselines.cpp)
Various versions are implemented:

- type == "burley"

    The isolated sheen component of the principled BRDF presented in "Physically Based Shading at Disney" by Brent Burley in the Physically-based Shading course at SIGGRAPH 2012.
    
    Scene format example:
    ```
    Material "sheenbaselines" "color Csheen" [1 1 1] "string type" ["burley"]
    ```
    
- type == "neubelt_pettineo"

    Sheen model presented in "Crafting a Next-Gen Material Pipeline for The Order: 1886" by Neubelt and Pettineo in the Physically-based Shading course at SIGGRAPH 2013.

    This is a modified version of a previous model from "A Microfacet-based BRDF Generator" by Ashikhmin et al. 2000.
    
    Scene format example:
    ```
    Material "sheenbaselines" "color Csheen" [1 1 1] "float alpha" [0.5] "string type" ["neubelt_pettineo"]
    ```

- type == "conty_kulla"

    Sheen model presented in "Production Friendly Microfacet Sheen BRDF" by Conty and Kulla in the Physically-based Shading course at SIGGRAPH 2017.
    
    Scene format example:
    ```
    Material "sheenbaselines" "color Csheen" [1 1 1] "float alpha" [0.5] "string type" ["conty_kulla"]
    ```

- type == "patry"

    Sheen model presented in "Samurai Shading in Ghost of Tsushima" by Jasmin Patry in the Physically-based Shading course at SIGGRAPH 2020.
    
    Scene format example:
    ```
    Material "sheenbaselines" "color Csheen" [1 1 1] "float alpha" [0.5] "string type" ["patry"]
    ```

## Test scenes

Two test scenes are included:

- The draped cloth example scene shown above: [`cloth.pbrt`](testscenes/cloth.pbrt)

- Directionally lit sphere [`sphere.pbrt`](testscenes/sphere.pbrt)

  The parameter `<angle>` in the scene description line
  ```
  Rotate <angle> 0 1 0
  ```
  can be used to change the angle of the directional light. The value is expected to be in degrees where `0` is front and `180` is back lighting.
