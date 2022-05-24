<p align="center">
  <img src="https://github.com/tizian/ltc-sheen/raw/master/images/teaser3.jpg" alt="Teaser" style="width: 100%;">
</p>

# LTC fitting code

This implementation reproduces the LTC fitting process for our new sheen BRDF. It is structured as a small standalone C++ implementation of different BRDF/LTC evaluation functionalities that are compiled into a python module `ltcsheen` via [pybind11](https://github.com/pybind/pybind11). This makes it convenient to then use Python to visualize the data and perform the fitting itself via [nonlinear optimization in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

## Compilation

1. Clone this repository _recursively_, i.e. including external submodules:
  ```
  git clone --recursive https://github.com/tizian/ltc-sheen.git
  ```
  
2. Navigate to the fitting subdirectory, e.g. on Unix:
  ```
  cd ltc-sheen/fitting
  ```

3. Follow the usual [CMake](https://cmake.org) build steps, e.g. on Unix:
  ```
  mkdir build
  cd build
  cmake ..
  make -j
  ```
  
Note: for this to work smootly, CMake needs to be able to detect your installation of Python on your `$PATH`.

## Explanations of scripts

The `python` subdirectory includes a number of scripts used for fitting and visualization purposes that should all be run directly from that directory.

* [`precompute_brdf_values_sheen_volume.py`](python/precompute_brdf_values_sheen_volume.py)

  The SGGX volume layer BRDF is inconvenient to fit directly because its evaluation is stochastic (and exhibits extremely high variance in some configurations). This script precomputes the relevant BRDF data for varying incident angle and roughness parameters. The results are written into two NumPy tensors (for data and hemispherical reflectance respectively):
  - `python/data/brdf_data_sheen_volume.npy`
  - `python/data/brdf_reflectance_sheen_volume.npy`

* [`fit_ltc_sheen_volume.py`](python/fit_ltc_sheen_volume.py)

  Runs the actual LTC fitting process for the (precomputed) SGGX volume layer BRDF. Outputs 32x32x3 dimensional LTC lookup tables in NumPy tensor and raw C++ files:
    - `python/data/ltc_table_sheen_volume.npy`
    - `python/data/ltc_table_sheen_volume.cpp`

* [`fit_ltc_sheen_approx.py`](python/fit_ltc_sheen_approx.py)
  
  Same as the previous script, but runs the LTC fitting on the analytic approximation of the BRDF instead. No precomputation is necessary in this case as the BRDF can be cheaply evaluated. Outputs corresponding LTC lookup tables:
    - `python/data/ltc_table_sheen_approx.npy`
    - `python/data/ltc_table_sheen_approx.cpp`

* [`brdf_plots_volume.py`](python/brdf_plots_volume.py)
  
  Displays an interactive plot (with varying incident angle and roughness inputs) to compare the fitted LTC BRDF against the ground truth SGGX volume layer BRDF.

* [`brdf_plots_approx.py`](python/brdf_plots_approx.py)

  Displays an interactive plot (with varying incident angle and roughness inputs) to compare the fitted LTC BRDF against the approximate analytic and ground truth SGGX volume layer BRDFs.

<br>

Note: A handful of external Python modules are used in these scripts that might not be included in your Python installation by default. They can all be installed via `pip install`:
  * [NumPy](https://pypi.org/project/numpy/) for tensor arithmetic.
  * [SciPy](https://pypi.org/project/scipy/) for nonlinear optimization / fitting.
  * [multiprocess](https://pypi.org/project/multiprocess/) for multithreaded computation.
  * [tqdm](https://pypi.org/project/tqdm/) for displaying progress bars.
  * [matplotlib](https://pypi.org/project/matplotlib/) for visualizing data.
  

## Third party code

The following external libraries are used:

- [pybind11](https://github.com/pybind/pybind11) for creating the Python bindings
- [pcg32](https://github.com/wjakob/pcg32) for random number generation
