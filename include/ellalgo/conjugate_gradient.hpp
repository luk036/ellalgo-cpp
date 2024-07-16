// conjugate_gradient.hpp

#ifndef CONJUGATE_GRADIENT_HPP
#define CONJUGATE_GRADIENT_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

class Vector {
  public:
    Vector(size_t size) : data(size, 0.0) {}
    Vector(const std::vector<double>& v) : data(v) {}
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    size_t size() const { return data.size(); }

    Vector& operator+=(const Vector& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] += rhs[i];
        return *this;
    }
    Vector& operator-=(const Vector& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] -= rhs[i];
        return *this;
    }
    Vector& operator*=(double scalar) {
        for (auto& val : data) val *= scalar;
        return *this;
    }

    double dot(const Vector& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    double norm() const { return std::sqrt(dot(*this)); }

  private:
    std::vector<double> data;
};

inline Vector operator+(Vector lhs, const Vector& rhs) {
    lhs += rhs;
    return lhs;
}

inline Vector operator-(Vector lhs, const Vector& rhs) {
    lhs -= rhs;
    return lhs;
}

inline Vector operator*(Vector v, double scalar) {
    v *= scalar;
    return v;
}

inline Vector operator*(double scalar, Vector v) { return v * scalar; }

class Matrix {
  public:
    Matrix(size_t rows, size_t cols) : data(rows, std::vector<double>(cols, 0.0)) {}
    std::vector<double>& operator[](size_t i) { return data[i]; }
    const std::vector<double>& operator[](size_t i) const { return data[i]; }
    size_t rows() const { return data.size(); }
    size_t cols() const { return data[0].size(); }

    Vector dot(const Vector& v) const {
        Vector result(rows());
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

inline Vector conjugate_gradient(const Matrix& A, const Vector& b, const Vector* x0 = nullptr,
                                 double tol = 1e-5, int max_iter = 1000) {
    size_t n = b.size();
    Vector x = x0 ? *x0 : Vector(n);

    Vector r = b - A.dot(x);
    Vector p = r;
    double r_norm_sq = r.dot(r);

    for (int i = 0; i < max_iter; ++i) {
        Vector Ap = A.dot(p);
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
