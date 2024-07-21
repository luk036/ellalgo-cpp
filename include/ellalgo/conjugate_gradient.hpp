// conjugate_gradient.hpp

#ifndef CONJUGATE_GRADIENT_HPP
#define CONJUGATE_GRADIENT_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

class Vector0 {
  public:
    Vector0(size_t size) : data(size, 0.0) {}
    Vector0(const std::vector<double>& v) : data(v) {}
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    size_t size() const { return data.size(); }

    Vector0& operator+=(const Vector0& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] += rhs[i];
        return *this;
    }
    Vector0& operator-=(const Vector0& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] -= rhs[i];
        return *this;
    }
    Vector0& operator*=(double scalar) {
        for (auto& val : data) val *= scalar;
        return *this;
    }

    double dot(const Vector0& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    double norm() const { return std::sqrt(dot(*this)); }

  private:
    std::vector<double> data;
};

inline Vector0 operator+(Vector0 lhs, const Vector0& rhs) {
    lhs += rhs;
    return lhs;
}

inline Vector0 operator-(Vector0 lhs, const Vector0& rhs) {
    lhs -= rhs;
    return lhs;
}

inline Vector0 operator*(Vector0 v, double scalar) {
    v *= scalar;
    return v;
}

inline Vector0 operator*(double scalar, Vector0 v) { return v * scalar; }

class Matrix0 {
  public:
    Matrix0(size_t rows, size_t cols) : data(rows, std::vector<double>(cols, 0.0)) {}
    std::vector<double>& operator[](size_t i) { return data[i]; }
    const std::vector<double>& operator[](size_t i) const { return data[i]; }
    size_t rows() const { return data.size(); }
    size_t cols() const { return data[0].size(); }

    Vector0 dot(const Vector0& v) const {
        Vector0 result(rows());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i] += data[i][j] * v[j];
            }
        }
        return result;
    }

  private:
    std::vector<std::vector<double>> data;
};

template <typename Matrix0, typename Vector0>
inline Vector0 conjugate_gradient(const Matrix0& A, const Vector0& b, const Vector0* x0 = nullptr,
                                  double tol = 1e-5, int max_iter = 1000) {
    size_t n = b.size();
    Vector0 x = x0 ? *x0 : Vector0(n);

    Vector0 r = b - A.dot(x);
    Vector0 p = r;
    double r_norm_sq = r.dot(r);

    for (int i = 0; i < max_iter; ++i) {
        Vector0 Ap = A.dot(p);
        double alpha = r_norm_sq / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        double r_norm_sq_new = r.dot(r);

        if (std::sqrt(r_norm_sq_new) < tol) {
            return x;
        }

        double beta = r_norm_sq_new / r_norm_sq;
        p = r + beta * p;
        r_norm_sq = r_norm_sq_new;
    }

    throw std::runtime_error("Conjugate Gradient did not converge after " + std::to_string(max_iter)
                             + " iterations");
}

#endif  // CONJUGATE_GRADIENT_HPP
