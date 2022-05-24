# This script computes the LTC fit of our sheen model based on the fiber-like
# SGGX volume layer with multiple scattering.

import sys, os, pathlib
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Make sure we can import the ltcsheen python package from here.
# This might need to be adjusted depending on your compilation setup and/or OS.
path = str(pathlib.Path(__file__).parent.resolve()) + "/../build"
sys.path.append(path)
import ltcsheen

def main():
    # Resolution of the final LTC lookup table we want to fit, for 2D inputs:
    # - incident angle cosine (`mu`)
    # - roughness (`alpha`)
    mu_res    = 32
    alpha_res = 32
    mus    = np.maximum(0.01, np.linspace(0, 1, mu_res))
    alphas = np.maximum(0.01, np.linspace(0, 1, alpha_res))

    # As part of the fitting process, we will compare the reference BRDF with the
    # one produced by the LTC approximation. Hence we will evaluate the BRDF many
    # times for different output directions, in particular for 32x32 directions that
    # are uniformly spaced in spherical coordinates. I.e.:
    # - `phi_o` runs from 0 - Pi radians. (Computing Pi - 2*Pi is redundant because
    #   the BRDF is isotropic.)
    # - `theta_o` runs from 0 to 0.5*Pi. (The lower hemisphere is zero as we only
    #   care about reflection.)
    theta_o_res = 32
    phi_o_res   = 32
    phi_os   = (np.arange(phi_o_res) + 0.5) / phi_o_res * np.pi
    theta_os = (np.arange(theta_o_res) + 0.5) / theta_o_res * 0.5*np.pi
    phi_o, theta_o = np.meshgrid(phi_os, theta_os)

    # The volumetric sheen BRDF is extremely noisy to evaluate for some of the input
    # configurations. We therefore use a precomputed table of the relevant BRDF data
    # for each of the 32x32 inputs.
    # See the script `precompute_brdf_values_sheen_volume.py` for details on how
    # these can be computed.
    brdf_reflectance = np.load("data/brdf_reflectance_sheen_volume.npy")
    brdf_data        = np.load("data/brdf_data_sheen_volume.npy")

    # ------------------------------------------------------------------------------

    # Objective function used for the LTC fitting process.
    def objective(x, brdf_values_ref, R_ref):
        # Note that we fit the values of the _inverse_ linear transform, i.e.
        # `M_i^{-1}` in Eq. (1) directly. This results in smoother lookup tables,
        # and thus smoother interpolation at low table resolutions.
        # We found it sufficient to only fit two values:
        # - `x[0]`, which is used for the two matrix entries `m_{1,1} = m_{2,2}`.
        # - `x[1]`, which is matrix entry `m_{1,3}`.
        # The remaining matrix elements are all zero, except for a constant
        # `m_{3,3} = 1`.
        # The LTC magnitude, i.e. `R_i` in Eq. (1), is also available as part of
        # the precomputed data and can be plugged in directly.
        M_inv = np.array([
            [x[0], 0.0,  x[1]],
            [0.0,  x[0], 0.0 ],
            [0.0,  0.0,  1.0]
        ])

        # Create LTC BRDF with these parameters, and evaluate for the given set of
        # outgoing directions.
        ltc = ltcsheen.LTCBrdf(R_ref, M_inv)
        ltc_values = ltc.eval_vectorized(0.0, 0.0, theta_o, phi_o)
        del ltc

        # For the actual objective, compare the computed LTC values against the
        # reference BRDF values we want to fit, with L3 loss.
        return np.sum(np.abs(ltc_values - brdf_values_ref)**3)

    # ------------------------------------------------------------------------------

    # Fitting process.
    print("* Fitting LTC coefficients ...")

    # Parameters of the "Nelder-Mead" optimizer. This is a downhill simplex solver
    # that works without having access to derivative information.
    tolerance = 1e-6
    maxIters  = 1000000

    ltc_table = np.zeros((mu_res, alpha_res, 3))

    # Go through each row in the LTC lookup table, i.e. the elevation angles.
    for mu_idx in tqdm(range(len(mus))):

        # Start with a simple initial guess.
        x_init = [1.0, 0.0]

        # Loop through roughness values from highest to lowest.
        for alpha_idx in range(len(alphas))[::-1]:
            # Lookup the precomputed BRDF reflectance. This can also be added
            # directly to the LTC lookup table as the magnitude parameter.
            R_ref = brdf_reflectance[mu_idx, alpha_idx]
            ltc_table[mu_idx, alpha_idx, 2] = R_ref

            # We simply skip this entry in case the reflectance is sufficiently
            # small. Conveniently, these are also the cases where the volume BRDF
            # data particularly noisy.
            if R_ref < 3e-6:
                continue

            # Lookup precomputed BRDF shape.
            brdf_values_ref = brdf_data[mu_idx, alpha_idx, :, :]

            # Run the nonlinear optimization.
            res = minimize(objective,
                           x_init,
                           method="Nelder-Mead",
                           args=(brdf_values_ref, R_ref,),
                           tol=tolerance,
                           options={"maxiter": maxIters})
            ltc_table[mu_idx, alpha_idx, :2] = res.x

            # Reuse the found solution as initial guess for next roughness value.
            # This improves the smoothness of the final table.
            x_init = res.x

    print("  done.")

    # ------------------------------------------------------------------------------

    # Save LTC tables as a `.npy` binary file.
    outname = "data/ltc_table_sheen_volume"
    np.save(outname, ltc_table)
    print("Saved as {}.npy".format(outname))

    # Also save formatted strings that can be copied into the provided pbrt-v3
    # C++ implementation.
    with open("{}.cpp".format(outname), "w") as f:
        v1 = ltc_table[:,:,0].T.flatten()
        v2 = ltc_table[:,:,1].T.flatten()
        v3 = brdf_reflectance.T.flatten()

        print("const Vector3f SheenLTC::_ltcParamTableVolume[32][32] = {", file=f)
        idx = 0
        for j in range(alpha_res):
            print("    {", file=f)

            for i in range(mu_res):
                if i % 3 == 0:
                    print("        ", end="", file=f)

                print("Vector3f({:.5f}, {:.5f}, {:.5f})".format(v1[idx],
                                                                v2[idx],
                                                                v3[idx]),
                      end="", file=f)
                print(", " if i < mu_res - 1 else "\n", end="", file=f)

                if i % 3 == 2:
                    print(file=f)

                idx += 1

            print("    }", end="", file=f)
            print("," if j < alpha_res - 1 else "", file=f)
        print("};", file=f)

    print("Saved as {}.cpp".format(outname))

    # ------------------------------------------------------------------------------

    # Plot a visualization of the LTC tables.
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    vmins = [0.0, -0.2, 0.0]
    vmaxs = [1.0, 0.0, 0.8]
    for i in range(3):
        im = ax[i].imshow(ltc_table[:,:,i],
                          cmap="turbo",
                          extent=[0, 1, 1, 0],
                          interpolation="nearest",
                          vmin=vmins[i],
                          vmax=vmaxs[i])
        cax = make_axes_locatable(ax[i]).append_axes("right", size="5%", pad=0.2)
        plt.colorbar(im, cax=cax)
        ax[i].set_xlabel(r"$\alpha$", size=16)
        ax[i].set_ylabel(r"$\cos\theta_i$", size=16)

    ax[0].set_title(r"$A_i$", size=16)
    ax[1].set_title(r"$B_i$", size=16)
    ax[2].set_title(r"$R_i$", size=16)
    plt.tight_layout()
    plt.suptitle("LTC coefficients (SGGX volume)", y=0.98, size=20)
    plt.show()

if __name__ == '__main__':
    main()
