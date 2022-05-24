#pragma once

namespace sggx {

/*
    From the reference implementaiton of

        "The SGGX Microflake Distribution"
        by Heitz et al. 2015.

    Used to evaluate and sample from the SGGX microflake phase function
    distribution.
*/

struct Ellipsoid {
    double xx, yy, zz, xy, xz, yz;

    Ellipsoid() {}

    static Ellipsoid from_fiber(const vec3 &t, double sigma) {
        double x = t.x();
        double y = t.y();
        double z = t.z();
        Ellipsoid S;
        S.xx = sigma*sigma*x*x + y*y + z*z;
        S.yy = sigma*sigma*y*y + x*x + z*z;
        S.zz = sigma*sigma*z*z + x*x + y*y;
        S.xy = sigma*sigma*x*y - x*y;
        S.xz = sigma*sigma*x*z - x*z;
        S.yz = sigma*sigma*y*z - y*z;
        return S;
    }
};

inline double ndf(const vec3 &wm, const Ellipsoid &S) {
    double detS = S.xx*S.yy*S.zz -
                  S.xx*S.yz*S.yz -
                  S.yy*S.xz*S.xz -
                  S.zz*S.xy*S.xy +
                  2.0*S.xy*S.xz*S.yz;

    double den = wm.x()*wm.x()*(S.yy*S.zz-S.yz*S.yz) +
                 wm.y()*wm.y()*(S.xx*S.zz-S.xz*S.xz) +
                 wm.z()*wm.z()*(S.xx*S.yy-S.xy*S.xy)
               + 2.0*(wm.x()*wm.y()*(S.xz*S.yz-S.zz*S.xy) +
                      wm.x()*wm.z()*(S.xy*S.yz-S.yy*S.xz) +
                      wm.y()*wm.z()*(S.xy*S.xz-S.xx*S.yz));

    return std::pow(std::abs(detS), 1.5) / (Pi*den*den);
}

inline double sigma(const vec3 &wi, const Ellipsoid &S) {
    double sigma2 = wi.x()*wi.x()*S.xx +
                    wi.y()*wi.y()*S.yy +
                    wi.z()*wi.z()*S.zz
                  + 2.0*(wi.x()*wi.y()*S.xy +
                         wi.x()*wi.z()*S.xz +
                         wi.y()*wi.z()*S.yz);

    return std::sqrt(std::max(0.0, sigma2));
}

inline double visible_ndf(const vec3 &wi, const vec3 &wm, const Ellipsoid &S) {
    return std::max(0.0, std::min(1.0, dot(wi, wm))) * ndf(wm, S) / sigma(wi, S);
}

inline vec3 sample_visible_ndf(const vec3 &wi, const Ellipsoid &S, double u1, double u2) {
    // Generate sample (u, v, w)
    double r   = std::sqrt(u1),
           phi = 2.0 * Pi * u2;

    double u = r * std::sin(phi),
           v = r * std::cos(phi),
           w = std::sqrt(1.0 - u*u - v*v);

    // Build orthonormal basis
    auto [wk, wj] = coordinate_system(wi);

    // Project S in this basis
    double S_kk = wk.x()*wk.x()*S.xx + wk.y()*wk.y()*S.yy + wk.z()*wk.z()*S.zz
                + 2.0 * (wk.x()*wk.y()*S.xy + wk.x()*wk.z()*S.xz + wk.y()*wk.z()*S.yz);
    double S_jj = wj.x()*wj.x()*S.xx + wj.y()*wj.y()*S.yy + wj.z()*wj.z()*S.zz
                + 2.0 * (wj.x()*wj.y()*S.xy + wj.x()*wj.z()*S.xz + wj.y()*wj.z()*S.yz);
    double S_ii = wi.x()*wi.x()*S.xx + wi.y()*wi.y()*S.yy + wi.z()*wi.z()*S.zz
                + 2.0 * (wi.x()*wi.y()*S.xy + wi.x()*wi.z()*S.xz + wi.y()*wi.z()*S.yz);
    double S_kj = wk.x()*wj.x()*S.xx + wk.y()*wj.y()*S.yy + wk.z()*wj.z()*S.zz
                + (wk.x()*wj.y() + wk.y()*wj.x())*S.xy
                + (wk.x()*wj.z() + wk.z()*wj.x())*S.xz
                + (wk.y()*wj.z() + wk.z()*wj.y())*S.yz;
    double S_ki = wk.x()*wi.x()*S.xx + wk.y()*wi.y()*S.yy + wk.z()*wi.z()*S.zz
                + (wk.x()*wi.y() + wk.y()*wi.x())*S.xy
                + (wk.x()*wi.z() + wk.z()*wi.x())*S.xz
                + (wk.y()*wi.z() + wk.z()*wi.y())*S.yz;
    double S_ji = wj.x()*wi.x()*S.xx + wj.y()*wi.y()*S.yy + wj.z()*wi.z()*S.zz
                + (wj.x()*wi.y() + wj.y()*wi.x())*S.xy
                + (wj.x()*wi.z() + wj.z()*wi.x())*S.xz
                + (wj.y()*wi.z() + wj.z()*wi.y())*S.yz;

    // Compute normal
    double sqrt_det_Skji = std::sqrt(std::abs(S_kk*S_jj*S_ii - S_kj*S_kj*S_ii - S_ki*S_ki*S_jj - S_ji*S_ji*S_kk + 2.0*S_kj*S_ki*S_ji));
    double inv_sqrt_Sii = 1.0 / std::sqrt(S_ii);
    double tmp = std::sqrt(S_jj*S_ii - S_ji*S_ji);
    vec3 Mk(sqrt_det_Skji / tmp, 0.0, 0.0);
    vec3 Mj(-inv_sqrt_Sii*(S_ki*S_ji - S_kj*S_ii) / tmp, inv_sqrt_Sii*tmp, 0);
    vec3 Mi(inv_sqrt_Sii*S_ki, inv_sqrt_Sii*S_ji, inv_sqrt_Sii*S_ii);
    vec3 wm_kji = normalize(u*Mk + v*Mj + w*Mi);

    // Rotate back to world basis
    return wm_kji.x() * wk + wm_kji.y() * wj + wm_kji.z() * wi;
}

inline double eval_phase_specular(const vec3 &wi, const vec3 &wo, const Ellipsoid &S) {
    vec3 wh = normalize(wi + wo);
    double s = sigma(wi, S);
    return s == 0.0 ? 0.0 : 0.25 * ndf(wh, S) / s;
}

inline vec3 sample_phase_specular(const vec3 &wi, const Ellipsoid &S, double u1, double u2) {
    vec3 wm = sample_visible_ndf(wi, S, u1, u2);
    return normalize(2.0*dot(wm, wi)*wm - wi);
}

};
