#pragma once

#include <vec3.h>
#include <cmath>
#include <sstream>

// Very basic 3x3 matrix implementation.

struct mat3 {
    double data[3][3];

    explicit mat3() {
        data[0][0] = data[1][1] = data[2][2] = 1.0;
        data[0][1] = data[0][2] = 0.0;
        data[1][0] = data[1][2] = 0.0;
        data[2][0] = data[2][1] = 0.0;
    }

    explicit mat3(const vec3 &c0, const vec3 &c1, const vec3 &c2) {
        data[0][0] = c0[0]; data[1][0] = c0[1]; data[2][0] = c0[2];
        data[0][1] = c1[0]; data[1][1] = c1[1]; data[2][1] = c1[2];
        data[0][2] = c2[0]; data[1][2] = c2[1]; data[2][2] = c2[2];
    }

    explicit mat3(double m00, double m01, double m02,
                  double m10, double m11, double m12,
                  double m20, double m21, double m22) {
        data[0][0] = m00; data[0][1] = m01; data[0][2] = m02;
        data[1][0] = m10; data[1][1] = m11; data[1][2] = m12;
        data[2][0] = m20; data[2][1] = m21; data[2][2] = m22;
    }

    explicit mat3(double m2[3][3]) {
        memcpy(data, m2, 9 * sizeof(double));
    }

    bool operator==(const mat3 &m2) const {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (data[i][j] != m2.data[i][j]) return false;
            }
        }
        return true;
    }

    bool operator!=(const mat3 &m2) const {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (data[i][j] != m2.data[i][j]) return true;
            }
        }
        return false;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "mat3[[" << data[0][0] << ", " << data[0][1] << ", " << data[0][2] << "]," << std::endl;
        out << "     [" << data[1][0] << ", " << data[1][1] << ", " << data[1][2] << "]," << std::endl;
        out << "     [" << data[2][0] << ", " << data[2][1] << ", " << data[2][2] << "]]" << std::endl;
        return out.str();
    }
};

std::ostream& operator<<(std::ostream& out, const mat3 &m) {
    out << "mat3[[" << m.data[0][0] << ", " << m.data[0][1] << ", " << m.data[0][2] << "]," << std::endl;
    out << "     [" << m.data[1][0] << ", " << m.data[1][1] << ", " << m.data[1][2] << "]," << std::endl;
    out << "     [" << m.data[2][0] << ", " << m.data[2][1] << ", " << m.data[2][2] << "]]" << std::endl;
    return out;
}

inline mat3 transpose(const mat3 &m) {
    return mat3(m.data[0][0], m.data[1][0], m.data[2][0],
                m.data[0][1], m.data[1][1], m.data[2][1],
                m.data[0][2], m.data[1][2], m.data[2][2]);
}

inline double det(const mat3 &m) {
    return m.data[0][0]*m.data[1][1]*m.data[2][2] +
           m.data[0][1]*m.data[1][2]*m.data[2][0] +
           m.data[0][2]*m.data[1][0]*m.data[2][1] -
           m.data[0][0]*m.data[1][2]*m.data[2][1] -
           m.data[0][1]*m.data[1][0]*m.data[2][2] -
           m.data[0][2]*m.data[1][1]*m.data[2][0];
}

inline mat3 inverse(const mat3 &m) {
    double inv_det = 1.0 / det(m);
    mat3 ret;
    ret.data[0][0] = inv_det * (m.data[1][1]*m.data[2][2] - m.data[1][2]*m.data[2][1]);
    ret.data[0][1] = inv_det * (m.data[0][2]*m.data[2][1] - m.data[0][1]*m.data[2][2]);
    ret.data[0][2] = inv_det * (m.data[0][1]*m.data[1][2] - m.data[0][2]*m.data[1][1]);
    ret.data[1][0] = inv_det * (m.data[1][2]*m.data[2][0] - m.data[1][0]*m.data[2][2]);
    ret.data[1][1] = inv_det * (m.data[0][0]*m.data[2][2] - m.data[0][2]*m.data[2][0]);
    ret.data[1][2] = inv_det * (m.data[0][2]*m.data[1][0] - m.data[0][0]*m.data[1][2]);
    ret.data[2][0] = inv_det * (m.data[1][0]*m.data[2][1] - m.data[1][1]*m.data[2][0]);
    ret.data[2][1] = inv_det * (m.data[0][1]*m.data[2][0] - m.data[0][0]*m.data[2][1]);
    ret.data[2][2] = inv_det * (m.data[0][0]*m.data[1][1] - m.data[0][1]*m.data[1][0]);
    return ret;
}

inline mat3 operator*(const mat3 &m1, const mat3 &m2) {
    mat3 ret;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ret.data[i][j] = m1.data[i][0] * m2.data[0][j] +
                             m1.data[i][1] * m2.data[1][j] +
                             m1.data[i][2] * m2.data[2][j];
        }
    }
    return ret;
}

inline vec3 operator*(const mat3 &m, const vec3 &v) {
    vec3 ret;
    for (int i = 0; i < 3; ++i) {
        ret[i] = v[0] * m.data[i][0] + v[1] * m.data[i][1] + v[2] * m.data[i][2];
    }
    return ret;
}
