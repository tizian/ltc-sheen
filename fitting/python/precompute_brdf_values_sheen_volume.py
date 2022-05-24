# The purpose of this script is to precompute BRDF reflectance data based on
# a random walk through a fiber-like SGGX volume.
#
# Important: this can easily take hourse to get good (noise-free) results. The
# particular settings below took approx. two days of computation on a
# 2021 MacBook Pro.
# Still, some BRDF configurations (low roughness and roughly perpendicular
# incidence angles) are still quite noisy, as very few sampled paths reflect
# back out of the volume in that case. Instead, they usually scatter a few times
# and leave on the bottom of the volume layer.
#
# For convenience, the result of the script is included in the repository
# directly, in form of `.npy` NumPy binary files.
#
# See the `n_passes` parameter below (on line 132) to adjust the computation
# time and output quality.

import sys, os, pathlib, multiprocess
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Make sure we can import the ltcsheen python package from here.
# This might need to be adjusted depending on your compilation setup and/or OS.
path = str(pathlib.Path(__file__).parent.resolve()) + "/../build"
sys.path.append(path)
import ltcsheen

def main():
    # Volume parameters used in the SGGX sheen model:
    # - How many bounces to simulate inside the volume, needs to be a relatively
    #   high number to account for multiple scattering.
    maxBounces = 16
    # - Use a non-absorptive medium.
    albedo = 1.0
    # - Volume density assuming a unit layer thickness.
    density = 1.0

    # Resolution of the final LTC lookup table we want to fit, for 2D inputs:
    # - incident angle cosine (`mu`)
    # - roughness (`alpha`)
    # In this script, we precompute the BRDF data at these same points.
    mu_res    = 32
    alpha_res = 32
    mus    = np.maximum(0.01, np.linspace(0, 1, mu_res))
    alphas = np.maximum(0.01, np.linspace(0, 1, alpha_res))

    # ------------------------------------------------------------------------------

    # As a first step, we compute the overall reflectance of the sheen for all 32x32
    # inputs. This can directly be used as the magnitude scale of the final LTC
    # distribution. I.e. `R_i` in Eq. (1).

    n_samples = 2000000 # Number of Monte Carlo samples used to estimate `R_i`.

    # Helper function for computing the reflectance. This is executed in a parallel
    # loop below.
    def sample_brdf(idx):
        # Convert 1D `idx` to 2D position in the lookup table.
        mu_idx    = idx %  mu_res
        alpha_idx = idx // mu_res
        mu    = mus[mu_idx]
        alpha = alphas[alpha_idx]

        # SGGX fiber cross section parameter, computed from the artist-specified
        # roughness value `alpha`.
        sigma = alpha*alpha

        # Create BRDF and a sampler to access random numbers.
        brdf = ltcsheen.VolumeSheenBrdf(maxBounces, albedo, density, sigma)
        sampler = ltcsheen.Sampler()
        sampler.seed(0)

        # Vectorized call that will sample `n_samples` times from the BRDF. We only
        # care about the returned sample weight (which is 1 or 0 in this specific
        # case), and not the actual returned directions.
        weight, _, _ = brdf.sample_vectorized(np.arccos(mu), 0.0,
                                              sampler=sampler, n_samples=n_samples)
        del brdf

        # Return average of all weights as an estimate of the reflectance.
        return np.mean(weight)

    # Run code above in parallel.
    print("* Precompute BRDF reflectance:")
    pool = multiprocess.Pool()
    p_result = list(tqdm(pool.imap(sample_brdf, range(0, mu_res*alpha_res)),
                         total=mu_res*alpha_res))
    brdf_reflectance = np.zeros((mu_res, alpha_res))

    # Store BRDF reflectance in a NumPy array.
    for idx in range(0, mu_res*alpha_res):
        mu_idx    = idx %  mu_res
        alpha_idx = idx // mu_res
        brdf_reflectance[mu_idx, alpha_idx] = p_result[idx]
    del pool

    # Save BRDF reflectance as a `.npy` binary file.
    outname = "data/brdf_reflectance_sheen_volume.npy"
    np.save(outname, brdf_reflectance)
    print("  Saved as {}".format(outname))
    print("")

    # ------------------------------------------------------------------------------

    # We now turn to tabulation of the BRDF shape, again for all 32x32 inputs. We
    # evaluate it for 32x32 outgoing directions, uniformly spaced in spherical
    # coordinates. I.e.:
    # - `phi_o` runs from 0 - Pi radians. (Computing Pi - 2*Pi is redundant because
    #   the BRDF is isotropic.)
    # - `theta_o` runs from 0 to 0.5*Pi. (The lower hemisphere is zero as we only
    #   care about reflection.)
    theta_o_res = 32
    phi_o_res   = 32
    phi_os   = (np.arange(phi_o_res)   + 0.5) / phi_o_res   * np.pi
    theta_os = (np.arange(theta_o_res) + 0.5) / theta_o_res * 0.5*np.pi
    phi_o, theta_o = np.meshgrid(phi_os, theta_os)

    # Instead of directly evaluating the BRDF directly for specific outgoing
    # directions, we found lower variance overall when _sampling_ from the BRDF and
    # then accumulating a sample histogram. Still, due to the random walk, many
    # samples are necessary in practice.
    n_evals = 50000
    n_samples = n_evals*theta_o_res*phi_o_res

    # Run multiple passes of the same computation to further decrease variance.
    # Reduce this in order to get quicker results (albeit with increased variance).
    # (Using larger `n_samples` directly results can otherwise exceed the available
    # memory.)
    n_passes = 100

    print("* Precompute BRDF data:")
    brdf_data = np.zeros((mu_res, alpha_res, theta_o_res, phi_o_res))

    for p in range(n_passes):
        print("  Pass {}/{} ..".format(p+1, n_passes))

        # Helper function for estimating the BRDF. This is executed in a parallel
        # loop below.
        def eval_brdf(idx):
            # Convert 1D `idx` to 2D position in the lookup table
            mu_idx    = idx %  mu_res
            alpha_idx = idx // mu_res
            mu    = mus[mu_idx]
            alpha = alphas[alpha_idx]

            # SGGX fiber cross section parameter, computed from the artist-specified
            # roughness value `alpha`.
            sigma = alpha*alpha

            # Again, create the BRDF and sampler.
            brdf = ltcsheen.VolumeSheenBrdf(maxBounces, albedo, density, sigma)
            sampler = ltcsheen.Sampler()

            # Each pass needs to use a different RNG seed.
            sampler.seed(p)

            if True:
                # Evaluate BRDF shape by sampling many times and accumulating a
                # sample histogram.
                weight, theta_o, phi_o = brdf.sample_vectorized(np.arccos(mu), 0.0,
                                                                sampler=sampler,
                                                                n_samples=n_samples)

                # Samples that end up between `phi_o=Pi` and `phi_o=2*Pi` are
                # mirrored s.t. the histogram is created only over the specified
                # `phi_o` range above.
                mirror = phi_o > np.pi
                phi_o[mirror] = 2*np.pi - phi_o[mirror]

                # Create a histogram of all valid (i.e. non-zero) samples. Note that
                # each sample is additionally weighted by the Jacobian determinant
                # of the spherical coordinate mapping, i.e. `1/sin(theta_o)`.
                mask = weight > 0.0
                values, _, _ = np.histogram2d(theta_o[mask], phi_o[mask],
                                              weights=1/np.sin(theta_o)[mask],
                                              bins=[theta_o_res, phi_o_res])

                # Apply additional scale factors s.t. the final sample histogram
                # agrees with the true BRDF values.
                values /= np.pi**2*n_samples
                values *= theta_o_res*phi_o_res
            else:
                # Only for reference, this would directly evaluate the BRDF at the
                # predetermined outgoing directions. However it is unused as it
                # tends to result in higher variance than the option above.
                phi_o, theta_o = np.meshgrid(phi_os, theta_os)
                values = brdf.eval_vectorized(np.arccos(mu), 0.0, theta_o, phi_o,
                                              n_evals=n_evals, sampler=sampler)

            # One last detail: the first row of the evaluated BRDF (i.e.
            # approximately prependicular incidence) receives samples spread out
            # over many `phi_o` bins, even though they all point (almost) into the
            # same direction. This is suboptimal and we can reduce variance here by
            # simply averaging.
            values[0, :] = np.mean(values[0, :])

            del brdf
            return values

        # Run code above in parallel.
        pool = multiprocess.Pool()
        p_result = list(tqdm(pool.imap(eval_brdf, range(0, mu_res*alpha_res)),
                             total=mu_res*alpha_res))

        # Store result in NumPy array.
        for idx in range(0, mu_res*alpha_res):
            mu_idx    = idx %  mu_res
            alpha_idx = idx // mu_res
            brdf_data[mu_idx, alpha_idx, :, :] += p_result[idx]
        del pool

    # Average BRDF data over all passes.
    brdf_data /= n_passes

    # Save BRDF data as a `.npy` binary file.
    outname = "data/brdf_data_sheen_volume.npy"
    np.save(outname, brdf_data)
    print("  Saved as {}".format(outname))

if __name__ == '__main__':
    main()
