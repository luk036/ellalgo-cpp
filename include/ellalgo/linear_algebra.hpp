// linear_algebra.hpp

#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <cmath>
#include <vector>

template <typename T> class Vector2 {
  public:
    using value_type = T;

    /**
     * Constructs a Vector2 with the given size, initializing all elements to the default value of
     * T.
     *
     * @param size The size of the Vector2.
     */
    Vector2(size_t size) : data(size, T{}) {}

    /**
     * Constructs a Vector2 from a std::vector.
     *
     * @param v The std::vector to initialize the Vector2 with.
     */
    Vector2(const std::vector<T>& v) : data(v) {}

    /**
     * Destroys the Vector2 object.
     *
     * @return None
     */
    ~Vector2() = default;

    /**
     * Provides access to the elements of the Vector2 object.
     *
     * @param i The index of the element to access.
     * @return A reference to the element at the specified index.
     */
    T& operator[](size_t i) { return data[i]; }

    /**
     * Provides constant access to the elements of the Vector2 object.
     *
     * @param i The index of the element to access.
     * @return A constant reference to the element at the specified index.
     */
    const T& operator[](size_t i) const { return data[i]; }

    /**
     * Returns the size of the Vector2 object.
     *
     * @return The size of the Vector2 object.
     */
    size_t size() const { return data.size(); }

    /**
     * Adds the elements of the given Vector2 to the corresponding elements of this Vector2.
     *
     * @param rhs The Vector2 to add to this Vector2.
     * @return A reference to this Vector2 after the addition.
     */
    Vector2& operator+=(const Vector2& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] += rhs[i];
        return *this;
    }

    /**
     * Subtracts the elements of the given Vector2 from the corresponding elements of this Vector2.
     *
     * @param rhs The Vector2 to subtract from this Vector2.
     * @return A reference to this Vector2 after the subtraction.
     */
    Vector2& operator-=(const Vector2& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] -= rhs[i];
        return *this;
    }

    /**
     * Multiplies each element of the Vector2 by the given scalar value.
     *
     * @param scalar The scalar value to multiply the Vector2 elements by.
     * @return A reference to this Vector2 after the multiplication.
     */
    Vector2& operator*=(T scalar) {
        for (auto& val : data) val *= scalar;
        return *this;
    }

    /**
     * Computes the dot product of this Vector2 with the given Vector2.
     *
     * @param other The other Vector2 to compute the dot product with.
     * @return The dot product of this Vector2 and the given Vector2.
     */
    T dot(const Vector2& other) const {
        T sum = T{};
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    /**
     * Computes the L2 norm (Euclidean length) of the Vector2.
     *
     * @return The L2 norm of the Vector2.
     */
    T norm() const { return std::sqrt(dot(*this)); }

  private:
    std::vector<T> data;
};

/**
 * Adds the elements of the given Vector2 to the corresponding elements of this Vector2.
 *
 * @param lhs The Vector2 to add the elements of the given Vector2 to.
 * @param rhs The Vector2 to add to the elements of the given Vector2.
 * @return The resulting Vector2 after the addition.
 */
template <typename T> inline Vector2<T> operator+(Vector2<T> lhs, const Vector2<T>& rhs) {
    lhs += rhs;
    return lhs;
}

/**
 * Subtracts the elements of the given Vector2 from the corresponding elements of this Vector2.
 *
 * @param lhs The Vector2 to subtract the elements of the given Vector2 from.
 * @param rhs The Vector2 to subtract from the elements of the given Vector2.
 * @return The resulting Vector2 after the subtraction.
 */
template <typename T> inline Vector2<T> operator-(Vector2<T> lhs, const Vector2<T>& rhs) {
    lhs -= rhs;
    return lhs;
}

/**
 * Multiplies the elements of the given Vector2 by the given scalar value.
 *
 * @param v The Vector2 to multiply by the scalar.
 * @param scalar The scalar value to multiply the Vector2 by.
 * @return The resulting Vector2 after the multiplication.
 */
template <typename T> inline Vector2<T> operator*(Vector2<T> v, T scalar) {
    v *= scalar;
    return v;
}

/**
 * Multiplies the given scalar value by the elements of the given Vector2.
 *
 * @param scalar The scalar value to multiply the Vector2 by.
 * @param v The Vector2 to multiply by the scalar.
 * @return The resulting Vector2 after the multiplication.
 */
template <typename T> inline Vector2<T> operator*(T scalar, Vector2<T> v) { return v * scalar; }

template <typename T> class Matrix2 {
  public:
    /**
     * Constructs a new Matrix2 with the specified number of rows and columns, initializing all
     * elements to the default value of type T.
     *
     * @param rows The number of rows in the Matrix2.
     * @param cols The number of columns in the Matrix2.
     */
    Matrix2(size_t rows, size_t cols) : data(rows, std::vector<T>(cols, T{})) {}

    /**
     * Returns a reference to the vector of elements at the specified row index.
     *
     * @param i The row index to access.
     * @return A reference to the vector of elements at the specified row index.
     */
    std::vector<T>& operator[](size_t i) { return data[i]; }

    /**
     * Returns a constant reference to the vector of elements at the specified row index.
     *
     * @param i The row index to access.
     * @return A constant reference to the vector of elements at the specified row index.
     */
    const std::vector<T>& operator[](size_t i) const { return data[i]; }

    /**
     * Returns the number of rows in the Matrix2.
     *
     * @return The number of rows in the Matrix2.
     */
    size_t rows() const { return data.size(); }

    /**
     * Returns the number of columns in the Matrix2.
     *
     * @return The number of columns in the Matrix2.
     */
    size_t cols() const { return data[0].size(); }

    /**
     * Multiplies the given Vector2 by the elements of this Matrix2.
     *
     * @param v The Vector2 to multiply by the elements of this Matrix2.
     * @return The resulting Vector2 after the multiplication.
     */
    Vector2<T> operator*(const Vector2<T>& v) const {
        Vector2<T> result(rows());
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
