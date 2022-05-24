#pragma once

#include <iostream>
#include <sstream>
#include <cmath>

// Numerical constants
constexpr double Pi       = 3.14159265358979323846;
constexpr double InvPi    = 0.31830988618379067154;
constexpr double Infinity = std::numeric_limits<double>::infinity();

// Refer to `pcg32` as `Sampler` in this codebase.
typedef pcg32 Sampler;

// Conversion from spherical coordinates to unit directions.
inline vec3 spherical_direction(double theta, double phi) {
    double sin_theta = std::sin(theta),
           cos_theta = std::cos(theta),
           sin_phi   = std::sin(phi),
           cos_phi   = std::cos(phi);
    return vec3(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    );
}

// Conversion from unit directions to spherical coordinates.
inline std::pair<double, double> spherical_coordinates(const vec3 &v) {
    double theta = std::acos(v.z()),
           phi   = std::atan2(v.y(), v.x());
    if (phi < 0.0) {
        phi += 2.0 * Pi;
    }
    return { theta, phi };
}

// From 'Building an Orthonormal Basis, Revisited' by Duff et al. (2017)
inline std::pair<vec3, vec3> coordinate_system(const vec3 &n) {
    double sign = n.z() >= 0.0 ? 1.0 : -1.0;
    const double a = -1.0 / (sign + n.z());
    const double b = n.x() * n.y() * a;
    vec3 t = vec3(1 + sign * n.x() * n.x() * a, sign * b, -sign * n.x());
    vec3 s = vec3(b, sign + n.y() * n.y() * a, -n.y());
    return { t, s };
}

// Cosine hemisphere sampling.
inline vec3 square_to_cosine_hemisphere(double u1, double u2) {
    double r   = std::sqrt(u1),
           phi = 2.0 * Pi * u2;

    vec3 v;
    v.x() = r * std::cos(phi);
    v.y() = r * std::sin(phi);
    v.z() = std::sqrt(1.0 - v.x() * v.x() - v.y() * v.y());
    return v;
}

// Cosine hemisphere sampling density.
inline float square_to_cosine_hemisphere_pdf(const vec3 &v) {
    return std::max(0.0, v.z()) * InvPi;
}

// String indentation.
inline std::string indent(const std::string &string, int amount = 2) {
    std::istringstream iss(string);
    std::ostringstream oss;
    std::string spacer(amount, ' ');
    bool firstLine = true;
    for (std::string line; std::getline(iss, line); ) {
        if (!firstLine) {
            oss << spacer;
        }
        oss << line;
        if (!iss.eof()) {
            oss << std::endl;
        }
        firstLine = false;
    }
    return oss.str();
}
