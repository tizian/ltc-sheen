#pragma once

#include <bsdf.h>
#include <sggx.h>

class VolumeSheenBrdf : public Brdf {
public:
    VolumeSheenBrdf(int maxBounces, double albedo, double density, double sigma, double thetaF=0.0, double phiF=0.0)
        : maxBounces(maxBounces), albedo(albedo), density(density), sigma(sigma) {
        fiberNormal = spherical_direction(thetaF, phiF);
    }

    double eval_impl(const vec3 &wi, const vec3 &wo, Sampler *sampler) const override {
        if (maxBounces == 0 || wi.z() <= 0.0 || wo.z() <= 0.0) {
            return 0.0;
        }

        sggx::Ellipsoid S = sggx::Ellipsoid::from_fiber(fiberNormal, sigma);

        vec3   dir    = -wi;  // Propagation direction.
        double depth  = 0.0;  // Depth inside the volume (Assume unit thickness. 1: top, 0: bottom).
        int    bounce = 0;    // Keep track of scattering order.
        double beta   = 1.0;  // Path throughput.

        double value = 0.0;

        // Volumetric random walk, with next event estimation at each scattering event.
        while (true) {
            // Extinction coefficient depends on projected microflake area.
            double sigma_t = density*sggx::sigma(-dir, S);

            // Sample distance, and update depth (along optical axis).
            double dist_sampled = (sigma_t == 0.0) ? Infinity : -std::log(1.0 - sampler->nextDouble()) / sigma_t;
            depth = depth + dist_sampled * -dir.z();

            // Possibly escape the layer.
            if (depth < 0.0 || depth > 1.0) break;

            // Next event estimation, i.e. connect to outgoing direction `wo`.
            if (wo.z() > 0.0) {
                double dist_along_wo = depth / std::abs(wo.z());
                double sigma_t_along_wo = density*sggx::sigma(-wo, S);

                double phase_function = sggx::eval_phase_specular(-dir, wo, S);
                value += beta * albedo * phase_function * std::exp(-dist_along_wo*sigma_t_along_wo);
            }

            // Scattering event.
            beta *= albedo;
            dir = sggx::sample_phase_specular(-dir, S, sampler->nextDouble(), sampler->nextDouble());

            bounce++;
            if (bounce >= maxBounces && maxBounces > 0) break;
        }

        return value;
    }

    std::pair<double, vec3> sample_impl(const vec3 &wi, Sampler *sampler) const override {
        if (maxBounces == 0 || wi.z() <= 0.0) {
            return { 0.0, vec3(0.0) };
        }

        sggx::Ellipsoid S = sggx::Ellipsoid::from_fiber(vec3(0.0, 0.0, 1.0), sigma);

        vec3   dir    = -wi;  // Propagation direction.
        double depth  = 0.0;  // Depth inside the volume (Assume unit thickness. 1: top, 0: bottom).
        int    bounce = 0;    // Keep track of scattering order.
        double beta   = 1.0;  // Path throughput.

        // Volumetric random walk, with next event estimation at each scattering event.
        while (true) {
            // Extinction coefficient depends on projected microflake area.
            double sigma_t = density*sggx::sigma(-dir, S);

            // Sample distance, and update depth (along optical axis).
            double dist_sampled = (sigma_t == 0.0) ? Infinity : -std::log(1.0 - sampler->nextDouble()) / sigma_t;
            depth = depth + dist_sampled * -dir.z();

            // Possibly escape the layer.
            if (depth < 0.0 || depth > 1.0) break;

            // Scattering event.
            beta *= albedo;
            dir = sggx::sample_phase_specular(-dir, S, sampler->nextDouble(), sampler->nextDouble());

            bounce++;
            if (bounce >= maxBounces && maxBounces > 0) break;
        }

        return { beta, dir };
    }

    std::string to_string() const override {
        std::ostringstream out;
        out << "VolumeSheenBrdf[" << std::endl;
        out << "  maxBounces = " << maxBounces << std::endl;
        out << "  albedo = " << albedo << std::endl;
        out << "  density = " << density << std::endl;
        out << "  sigma = " << sigma << std::endl;
        out << "]";
        return out.str();
    }

public:
    int maxBounces;
    double albedo, density, sigma;
    vec3 fiberNormal;
};
