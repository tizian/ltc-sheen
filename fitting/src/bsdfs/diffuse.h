#pragma once

#include <bsdf.h>

/* Lambertian diffuse BRDF.
   Only for testing purposes. */

class DiffuseBrdf : public Brdf {
public:
    DiffuseBrdf(double reflectance)
        : reflectance(reflectance) {}

    double eval_impl(const vec3 &wi, const vec3 &wo, Sampler *sampler) const override {
        return reflectance * InvPi * wo.z();
    }

    std::pair<double, vec3> sample_impl(const vec3 &wi, Sampler *sampler) const override {
        vec3 wo = square_to_cosine_hemisphere(sampler->nextDouble(), sampler->nextDouble());
        return { reflectance, wo };
    }

    std::string to_string() const override {
        std::ostringstream out;
        out << "DiffuseBrdf[" << std::endl;
        out << "  reflectance = " << reflectance << std::endl;
        out << "]";
        return out.str();
    }

public:
    double reflectance;
};
