// linear_algebra.hpp

#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <cmath>
#include <vector>

template <typename T> class Vector {
  public:
    using value_type = T;

    Vector(size_t size) : data(size, T{}) {}
    Vector(const std::vector<T>& v) : data(v) {}
    ~Vector() = default;
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    size_t size() const { return data.size(); }

    Vector& operator+=(const Vector& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] += rhs[i];
        return *this;
    }
    Vector& operator-=(const Vector& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] -= rhs[i];
        return *this;
    }
    Vector& operator*=(T scalar) {
        for (auto& val : data) val *= scalar;
        return *this;
    }

    T dot(const Vector& other) const {
        T sum = T{};
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    T norm() const { return std::sqrt(dot(*this)); }

  private:
    std::vector<T> data;
};

template <typename T> inline Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs) {
    lhs += rhs;
    return lhs;
}

template <typename T> inline Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <typename T> inline Vector<T> operator*(Vector<T> v, T scalar) {
    v *= scalar;
    return v;
}

template <typename T> inline Vector<T> operator*(T scalar, Vector<T> v) { return v * scalar; }

template <typename T> class Matrix {
  public:
    Matrix(size_t rows, size_t cols) : data(rows, std::vector<T>(cols, T{})) {}
    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }
    size_t rows() const { return data.size(); }
    size_t cols() const { return data[0].size(); }

    Vector<T> operator*(const Vector<T>& v) const {
        Vector<T> result(rows());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i] += data[i][j] * v[j];
            }
        }
        return result;
    }

  private:
    std::vector<std::vector<T>> data;
};

#endif  // LINEAR_ALGEBRA_HPP
