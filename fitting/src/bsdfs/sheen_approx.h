#pragma once

#include <bsdf.h>
#include <sggx.h>

/*
    Analytic approximation of a volumetric scattering layer defined in
    `sheen_volume.h`.

    - Single-scattering is computed exactly.

    - Multiple-scattering uses an empirical approximation that is a very
      close fit to simulated multiple-scattering using a random walk.
*/

class ApproxSheenBrdf : public Brdf {
public:
    ApproxSheenBrdf(double alpha)
        : alpha(alpha) {}

    double eval_impl(const vec3 &wi, const vec3 &wo, Sampler *sampler) const override {
        const double density   = 1.0;
        const double thickness = 1.0;
        const double ss_albedo = 1.0;

        double cos_theta_i = wi.z(),
               cos_theta_o = wo.z();
        if (cos_theta_i <= 0.0 || cos_theta_o < 0.0) {
            return 0.0;
        }

        // Reparameterize alpha -> SGGX fiber cross section.
        double sigma = alpha * alpha;

        // Single scattering.
        sggx::Ellipsoid S = sggx::Ellipsoid::from_fiber(vec3(0.0, 0.0, 1.0), sigma);
        double lambda_wi = sggx::sigma(wi, S) / cos_theta_i,
               lambda_wo = sggx::sigma(wo, S) / cos_theta_o;

        double tmp0 = density*(lambda_wi + lambda_wo),
               tmp1 = 1.0 - std::exp(-thickness*tmp0);
        double tmp = tmp1 / (std::abs(cos_theta_i) * tmp0);

        double sigma_t_along_wi = density*sggx::sigma(wi, S),
               phase_function   = sggx::eval_phase_specular(wi, wo, S);

        double value = ss_albedo * phase_function * sigma_t_along_wi * tmp;

        if (!std::isfinite(value)) {
            /* Expressions `lambda_wi` and `lambda_wo` above tend
               towards infinity for grazing angles. */
            return 0.0;
        }

        // Multiple-scattering.
        auto [m, b] = mb(sigma);
        double ms = std::exp(b - m * (std::sqrt(cos_theta_o * cos_theta_o + cos_theta_i * cos_theta_i)));
        ms *= Ft(sigma, cos_theta_o) * Ft(sigma, cos_theta_i);
        value += ms * cos_theta_o;

        return value;
    }

    std::pair<double, vec3> sample_impl(const vec3 &wi, Sampler *sampler) const override {
        throw std::runtime_error("\"ApproxSheenBrdf::sample_impl\" not implemented.");
    }

    std::string to_string() const override {
        std::ostringstream out;
        out << "ApproxSheenBrdf[" << std::endl;
        out << "  alpha = " << alpha << std::endl;
        out << "]";
        return out.str();
    }

private:
    double Ft(double sigma, double c) const {
        double b = std::max(.55, 0.44 + 0.42 * sigma);
        return b + (1 - b) * (1 - std::pow(1 - c, 12.0));
    }

    std::pair<double, double> mb(double sigma) const {
        /* Bicubic bspline interpolation of fitted `m` and `b` vals, sampled
           for sigma = (0, 0.1, 0.2, ... 1.0).
           The first and last values are extrapolated such that the bspline
           interpolates the ends. */
        constexpr double mv0 = 2.0 * 20.0 - 10.2,
                         mv12 = 2.0 * 0.95 - 1.1;
        constexpr double mvals[13] = {mv0, 20.0, 10.2, 6.14, 4.30, 3.30, 2.55, 2.02, 1.63, 1.33, 1.1, 0.95, mv12};
        constexpr double bv0 = 2.0 * 3.7 - 2.23,
                         bv12 = 2.0 * -1.65 + 1.55;
        constexpr double bvals[13] = {bv0, 3.7, 2.23, 1.06, 0.293, -0.212, -0.641, -0.952, -1.2, -1.4, -1.55, -1.65, bv12};

        double r = sigma * 10.0;
        double ri = std::min(std::floor(r), 9.0);
        double m0 = mvals[int(ri)],
               m1 = mvals[int(ri) + 1],
               m2 = mvals[int(ri) + 2],
               m3 = mvals[int(ri) + 3];
        double b0 = bvals[int(ri)],
               b1 = bvals[int(ri) + 1],
               b2 = bvals[int(ri) + 2],
               b3 = bvals[int(ri) + 3];
        double y = r - ri,
               y2 = y * y,
               y3 = y2 * y;
        double B0 = -y + 2.0 * y2 - y3,
               B1 = 2.0 - 5.0 * y2 + 3.0 * y3,
               B2 = y + 4.0 * y2 - 3.0 * y3,
               B3 = -y2 + y3;
        double m = 0.5 * (B0 * m0 + B1 * m1 + B2 * m2 + B3 * m3);
        double b = 0.5 * (B0 * b0 + B1 * b1 + B2 * b2 + B3 * b3);

        return { m, b };
    }

public:
    double alpha;
};
