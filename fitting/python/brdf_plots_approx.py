# This script generates interactive BRDF plots to compare our LTC fit vs. the
# original fiber-like SGGX volume layer BRDF, as well as the analytic
# approximation used for one of the LTC fits.

import sys, os, pathlib, multiprocess
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
    # Input parameter space where we want to plot.
    mu_res    = 32
    alpha_res = 32
    mus    = np.maximum(0.01, np.linspace(0, 1, mu_res))
    alphas = np.maximum(0.01, np.linspace(0, 1, alpha_res))

    # Range of the outgoing hemisphere where the BRDF will be evaluated.
    # - From 0 - Pi in phi (Pi - 2*Pi is a mirror copy)
    # - From 0 - 0.5*Pi in theta (lower hemisphere is zero)
    theta_o_res = 32
    phi_o_res   = 32
    phi_os   = (np.arange(phi_o_res) + 0.5) / phi_o_res * np.pi
    theta_os = (np.arange(theta_o_res) + 0.5) / theta_o_res * 0.5*np.pi
    phi_o, theta_o = np.meshgrid(phi_os, theta_os)

    # ------------------------------------------------------------------------------

    # Load the precomputed BRDF values of the volumetric sheen model.
    brdf_data_sheen_volume = np.load("data/brdf_data_sheen_volume.npy")

    # Precompute the corresponding approx. and the LTC fit for all inputs.
    ltc_table_sheen_approx = np.load("data/ltc_table_sheen_approx.npy")
    brdf_data_ltc          = np.zeros((mu_res, alpha_res, theta_o_res, phi_o_res))
    brdf_data_sheen_approx = np.zeros((mu_res, alpha_res, theta_o_res, phi_o_res))

    def eval_brdf(idx):
        mu_idx    = idx %  mu_res
        alpha_idx = idx // mu_res
        mu    = mus[mu_idx]
        alpha = alphas[alpha_idx]

        # Evaluate the analytic sheen approximation.
        brdf = ltcsheen.ApproxSheenBrdf(alpha)
        values_sheen_approx = brdf.eval_vectorized(np.arccos(mu), 0.0, theta_o, phi_o)
        del brdf

        # Evaluate the corresponding LTC fit.
        A = ltc_table_sheen_approx[mu_idx, alpha_idx, 0]
        B = ltc_table_sheen_approx[mu_idx, alpha_idx, 1]
        M_inv = np.array([
            [A,   0.0, B],
            [0.0, A,   0.0],
            [0.0, 0.0, 1.0]
        ])

        R = ltc_table_sheen_approx[mu_idx, alpha_idx, 2]
        brdf = ltcsheen.LTCBrdf(R, M_inv)
        values_ltc = brdf.eval_vectorized(np.arccos(mu), 0.0, theta_o, phi_o)
        del brdf

        return values_sheen_approx, values_ltc

    pool = multiprocess.Pool()
    parallel_result = list(pool.imap(eval_brdf, range(0, mu_res*alpha_res)))
    for idx in range(0, mu_res*alpha_res):
        mu_idx    = idx %  mu_res
        alpha_idx = idx // mu_res
        brdf_data_sheen_approx[mu_idx, alpha_idx, :, :] = parallel_result[idx][0]
        brdf_data_ltc[mu_idx, alpha_idx, :, :] = parallel_result[idx][1]
    del pool

    # ------------------------------------------------------------------------------

    # Interactive plot of the three models next to each other.
    # Two sliders at the bottom control the inputs `theta` and `alpha`.

    fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
    im0 = ax[0].imshow(brdf_data_sheen_volume[0, 0, :, :],
                       cmap="turbo",
                       extent=[0, 2*np.pi, np.pi/2, 0],
                       interpolation="nearest")
    cbar_ax0 = make_axes_locatable(ax[0]).append_axes("right", size="2.5%", pad=0.2)
    plt.colorbar(im0, cax=cbar_ax0)

    im1 = ax[1].imshow(brdf_data_sheen_approx[0, 0, :, :],
                       cmap="turbo",
                       extent=[0, 2*np.pi, np.pi/2, 0],
                       interpolation="nearest")
    cbar_ax1 = make_axes_locatable(ax[1]).append_axes("right", size="2.5%", pad=0.2)
    plt.colorbar(im1, cax=cbar_ax1)

    im2 = ax[2].imshow(brdf_data_ltc[0, 0, :, :],
                       cmap="turbo",
                       extent=[0, 2*np.pi, np.pi/2, 0],
                       interpolation="nearest")
    cbar_ax2 = make_axes_locatable(ax[2]).append_axes("right", size="2.5%", pad=0.2)
    plt.colorbar(im2, cax=cbar_ax2)

    for ax_ in ax:
        ax_.set_xlabel(r"$\phi_o$", size=16)
        ax_.set_ylabel(r"$\theta_o$", size=16)
        ax_.set_xticks(np.arange(5)*0.5*np.pi)
        ax_.set_xticklabels(["0˚", "90˚", "180˚", "270˚", "360˚"])
        ax_.set_yticks([0, 0.25*np.pi, 0.5*np.pi])
        ax_.set_yticklabels(["0˚", "45˚", "90˚"])

    ax[0].set_title("Sheen (SGGX volume)", size=15)
    ax[1].set_title("Sheen (analytic approximation)", size=15)
    ax[2].set_title("LTC fit", size=15)

    mu_ax    = plt.axes([0.2, 0.025, 0.2, 0.025])
    mu_sl    = matplotlib.widgets.Slider(mu_ax, "theta:", 0.0, 90.0,
                                         valinit=45.0, valstep=1.0)
    alpha_ax = plt.axes([0.6, 0.025, 0.2, 0.025])
    alpha_sl = matplotlib.widgets.Slider(alpha_ax, "alpha:", 0.0, 1.0,
                                         valinit=0.5, valstep=0.01)

    def update(val):
        mu_idx    = np.abs(mus - np.cos(np.radians(mu_sl.val))).argmin()
        alpha_idx = np.abs(alphas - alpha_sl.val).argmin()

        tmp = brdf_data_sheen_volume[mu_idx, alpha_idx, :, :]
        data0 = np.hstack([tmp, np.fliplr(tmp)])
        im0.set_array(data0)
        im0.set(clim=(0, np.max(data0[1:, :])))
        plt.colorbar(im0, cax=cbar_ax0)

        tmp = brdf_data_sheen_approx[mu_idx, alpha_idx, :, :]
        data1 = np.hstack([tmp, np.fliplr(tmp)])
        im1.set_array(data1)
        im1.set(clim=(0, np.max(data0[1:, :])))
        plt.colorbar(im1, cax=cbar_ax1)

        tmp = brdf_data_ltc[mu_idx, alpha_idx, :, :]
        data2 = np.hstack([tmp, np.fliplr(tmp)])
        im2.set_array(data2)
        im2.set(clim=(0, np.max(data0[1:, :])))
        plt.colorbar(im2, cax=cbar_ax2)
    update(0)

    alpha_sl.on_changed(update)
    mu_sl.on_changed(update)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

if __name__ == '__main__':
    main()
