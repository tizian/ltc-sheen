#pragma once

#include <cmath>
#include <sstream>

// Very basic 3d vector implementation.

struct vec3 {
    double data[3];

    explicit vec3() {
        data[0] = 0.0;
        data[1] = 0.0;
        data[2] = 0.0;
    }

    explicit vec3(double x) {
        data[0] = x;
        data[1] = x;
        data[2] = x;
    }

    explicit vec3(double x, double y, double z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    double x() const { return data[0]; }
    double y() const { return data[1]; }
    double z() const { return data[2]; }

    double &x() { return data[0]; }
    double &y() { return data[1]; }
    double &z() { return data[2]; }

    bool operator==(const vec3 &v) const {
        return x() == v.x() && y() == v.y() && z() == v.z();
    }

    bool operator!=(const vec3 &v) const {
        return x() != v.x() || y() != v.y() || z() != v.z();
    }

    vec3 operator+(const vec3 &v) const {
        return vec3(x() + v.x(), y() + v.y(), z() + v.z());
    }

    vec3 &operator+=(const vec3 &v) {
        x() += v.x(); y() += v.y(); z() += v.z();
        return *this;
    }

    vec3 operator-(const vec3 &v) const {
        return vec3(x() - v.x(), y() - v.y(), z() - v.z());
    }

    vec3 &operator-=(const vec3 &v) {
        x() -= v.x(); y() -= v.y(); z() -= v.z();
        return *this;
    }

    vec3 &operator*=(double s) {
        x() *= s; y() *= s; z() *= s;
        return *this;
    }

    vec3 operator/(double s) const {
        return vec3(x() / s, y() / s, z() / s);
    }

    vec3 &operator/=(double s) {
        x() /= s; y() /= s; z() /= s;
        return *this;
    }

    vec3 operator-() const {
        return vec3(-x(), -y(), -z());
    }

    double operator[](int i) const {
        return data[i];
    }

    double &operator[](int i) {
        return data[i];
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "vec3[" << x() << ", " << y() << ", " << z() << "]";
        return out.str();
    }
};

std::ostream& operator<<(std::ostream& out, const vec3 &v) {
    out << "vec3[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
    return out;
}

inline vec3 min(const vec3 &v1, const vec3 &v2) {
    return vec3(std::min(v1.x(), v2.x()), std::min(v1.y(), v2.y()), std::min(v1.z(), v2.z()));
}

inline vec3 max(const vec3 &v1, const vec3 &v2) {
    return vec3(std::max(v1.x(), v2.x()), std::max(v1.y(), v2.y()), std::max(v1.z(), v2.z()));
}

inline vec3 abs(const vec3 &v) {
    return vec3(std::abs(v.x()), std::abs(v.y()), std::abs(v.z()));
}

inline double dot(const vec3 &v1, const vec3 &v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

inline double abs_dot(const vec3 &v1, const vec3 &v2) {
    return std::abs(dot(v1, v2));
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3((v1.y() * v2.z()) - (v1.z() * v2.y()),
                (v1.z() * v2.x()) - (v1.x() * v2.z()),
                (v1.x() * v2.y()) - (v1.y() * v2.x()));
}

inline double squared_norm(const vec3 &v) {
    return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}

inline double norm(const vec3 &v) {
    return std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

inline vec3 normalize(const vec3 &v) {
    double l = norm(v);
    return vec3(v.x() / l, v.y() / l, v.z() / l);
}

inline vec3 operator*(double s, const vec3 &v) {
    return vec3(s * v.x(), s * v.y(), s * v.z());
}

inline vec3 operator*(const vec3 &v, double s) {
    return vec3(s * v.x(), s * v.y(), s * v.z());
}
