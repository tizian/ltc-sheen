#pragma once

#include <global.h>

// Base class for BRDFs.

class Brdf {
public:
    virtual ~Brdf() {}

    // Evaluates the BRDF, including the cosine foreshortening factor, i.e. f_s * cos(theta_o).
    double eval(double theta_i, double phi_i, double theta_o, double phi_o,
                int n_evals = 1, Sampler *sampler = nullptr) const {
        // Convert to unit directions
        vec3 wi = spherical_direction(theta_i, phi_i),
             wo = spherical_direction(theta_o, phi_o);
        // Backfacing?
        if (wi.z() < 0.0 || wo.z() < 0.0) {
            return 0.0;
        }

        double value = 0.0;
        for (int i = 0; i < n_evals; ++i) {
            value += eval_impl(wi, wo, sampler);
        }
        return value / double(n_evals);
    }

    // Samples from the BRDF.
    std::tuple<double, double, double> sample(double theta_i, double phi_i, Sampler *sampler) const {
        // Convert to unit direction
        vec3 wi = spherical_direction(theta_i, phi_i);
        if (wi.z() < 0.0) {
            return { 0.0, 0.0, 0.0 };
        }

        auto [weight, wo] = sample_impl(wi, sampler);
        if (wo.z() < 0.0) {
            return { 0.0, 0.0, 0.0 };
        }

        // Convert from unit direction
        auto [theta_o, phi_o] = spherical_coordinates(wo);
        return { weight, theta_o, phi_o };
    }

    // Virtual implementation of `eval`.
    virtual double eval_impl(const vec3 &wi, const vec3 &wo, Sampler *sampler) const = 0;

    // Virtual implementation of `sample`.
    virtual std::pair<double, vec3> sample_impl(const vec3 &wi, Sampler *sampler) const = 0;

    // String description of the BRDF.
    virtual std::string to_string() const = 0;
};
