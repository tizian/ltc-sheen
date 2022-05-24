#pragma once

#include <bsdf.h>

/*
    Linearly transformed cosine BRDF, see

        "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
        by Heitz et al. 2016

    for more details.
*/

class LTCBrdf : public Brdf {
public:
    LTCBrdf(double magnitude, const mat3 &inv_transform)
        : magnitude(magnitude), inv_transform(inv_transform) {}

    double eval_impl(const vec3 &/* wi */, const vec3 &wo, Sampler *sampler) const override {
        double determinant = abs(det(inv_transform));

        vec3 wo_original = inv_transform * wo;
        double length = norm(wo_original);
        wo_original /= length;

        double D = square_to_cosine_hemisphere_pdf(wo_original);
        double jacobian = determinant / (length*length*length);

        return magnitude * D * jacobian;
    }

    std::pair<double, vec3> sample_impl(const vec3 &wi, Sampler *sampler) const override {
        vec3 wo_original = square_to_cosine_hemisphere(sampler->nextDouble(),
                                                       sampler->nextDouble());
        mat3 transform = inverse(inv_transform);
        vec3 wo = normalize(transform * wo_original);

        return { magnitude, wo };
    }

    std::string to_string() const override {
        std::ostringstream out;
        out << "LTCBrdf[" << std::endl;
        out << "  magnitude = " << magnitude << std::endl;
        out << "  inv_transform = " << indent(inv_transform.to_string(), 14) << std::endl;
        out << "]";
        return out.str();
    }

public:
    double magnitude;
    mat3 inv_transform;
};
