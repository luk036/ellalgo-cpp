/**
 * @file conjugate_gradient.hpp
 * @brief Conjugate gradient method for solving linear systems
 *
 * This header provides classes and functions for solving linear systems using
 * the conjugate gradient method. It includes a simple Vector and Matrix class
 * for demonstration purposes.
 */

#ifndef CONJUGATE_GRADIENT_HPP
#define CONJUGATE_GRADIENT_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief A simple vector class for conjugate gradient calculations
 *
 * This class provides basic vector operations needed for the conjugate gradient
 * method, including addition, subtraction, scalar multiplication, dot product,
 * and norm calculation.
 */
class Vector0 {
  public:
    /**
 * @brief Construct a new Vector0 object with given size
 *
 * @param[in] size The size of the vector
 */
Vector0(size_t size) : data(size, 0.0) {}

    /**
     * @brief Construct a new Vector0 from a std::vector
     *
     * @param[in] v The std::vector to copy from
     */
    Vector0(const std::vector<double>& v) : data(v) {}

    /**
     * @brief Access element at index i
     *
     * @param[in] i Index of the element
     * @return Reference to the element
     */
    double& operator[](size_t i) { return data[i]; }

    /**
     * @brief Access element at index i (const version)
     *
     * @param[in] i Index of the element
     * @return Const reference to the element
     */
    const double& operator[](size_t i) const { return data[i]; }

    /**
     * @brief Get the size of the vector
     *
     * @return Size of the vector
     */
    size_t size() const { return data.size(); }

    /**
     * @brief Add another vector to this vector
     *
     * @param[in] rhs The vector to add
     * @return Reference to this vector after addition
     */
    Vector0& operator+=(const Vector0& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] += rhs[i];
        return *this;
    }
    /**
     * @brief Subtract another vector from this vector
     *
     * @param[in] rhs The vector to subtract
     * @return Reference to this vector after subtraction
     */
    Vector0& operator-=(const Vector0& rhs) {
        for (size_t i = 0; i < size(); ++i) data[i] -= rhs[i];
        return *this;
    }
    /**
     * @brief Multiply this vector by a scalar
     *
     * @param[in] scalar The scalar value to multiply
     * @return Reference to this vector after multiplication
     */
    Vector0& operator*=(double scalar) {
        for (auto& val : data) val *= scalar;
        return *this;
    }

    /**
     * @brief Compute the dot product with another vector
     *
     * @param[in] other The other vector
     * @return The dot product result
     */
    double dot(const Vector0& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) sum += data[i] * other[i];
        return sum;
    }

    /**
     * @brief Compute the L2 norm of the vector
     *
     * @return The L2 norm (Euclidean length)
     */
    double norm() const { return std::sqrt(dot(*this)); }

  private:
    std::vector<double> data;
};

/**
 * @brief Add two vectors
 *
 * @param[in] lhs Left-hand side vector
 * @param[in] rhs Right-hand side vector
 * @return The resulting vector
 */
inline Vector0 operator+(Vector0 lhs, const Vector0& rhs) {
    lhs += rhs;
    return lhs;
}

/**
 * @brief Subtract two vectors
 *
 * @param[in] lhs Left-hand side vector
 * @param[in] rhs Right-hand side vector
 * @return The resulting vector
 */
inline Vector0 operator-(Vector0 lhs, const Vector0& rhs) {
    lhs -= rhs;
    return lhs;
}

/**
 * @brief Multiply a vector by a scalar
 *
 * @param[in] v The vector to multiply
 * @param[in] scalar The scalar value
 * @return The resulting vector
 */
inline Vector0 operator*(Vector0 v, double scalar) {
    v *= scalar;
    return v;
}

/**
 * @brief Multiply a scalar by a vector
 *
 * @param[in] scalar The scalar value
 * @param[in] v The vector to multiply
 * @return The resulting vector
 */
inline Vector0 operator*(double scalar, Vector0 v) { return v * scalar; }

/**
 * @brief A simple matrix class for conjugate gradient calculations
 *
 * This class provides basic matrix operations needed for the conjugate gradient
 * method, including matrix-vector multiplication.
 */
class Matrix0 {
  public:
    /**
     * @brief Construct a new Matrix0 object
     *
     * @param[in] rows Number of rows
     * @param[in] cols Number of columns
     */
    Matrix0(size_t rows, size_t cols) : data(rows, std::vector<double>(cols, 0.0)) {}

    /**
     * @brief Access row i of the matrix
     *
     * @param[in] i Row index
     * @return Reference to the row vector
     */
    std::vector<double>& operator[](size_t i) { return data[i]; }

    /**
     * @brief Access row i of the matrix (const version)
     *
     * @param[in] i Row index
     * @return Const reference to the row vector
     */
    const std::vector<double>& operator[](size_t i) const { return data[i]; }

    /**
     * @brief Get the number of rows
     *
     * @return Number of rows
     */
    size_t rows() const { return data.size(); }

    /**
     * @brief Get the number of columns
     *
     * @return Number of columns
     */
    size_t cols() const { return data[0].size(); }

    /**
     * @brief Multiply the matrix by a vector
     *
     * @param[in] v The vector to multiply
     * @return The resulting vector
     */
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

/**
 * Solves the linear system Ax = b using the conjugate gradient method.
 *
 * @tparam Matrix0 The matrix type, which must support matrix-vector multiplication.
 * @tparam Vector0 The vector type, which must support vector operations.
 * @param A The matrix A in the linear system Ax = b.
 * @param b The right-hand side vector b in the linear system Ax = b.
 * @param x0 An optional initial guess for the solution vector.
 * @param tol The tolerance for the residual norm, used as the stopping criterion.
 * @param max_iter The maximum number of iterations to perform.
 * @return The solution vector.
 * @throws std::runtime_error if the conjugate gradient method does not converge after the maximum
 * number of iterations.
 */
template <typename Matrix0, typename Vector0>
inline Vector0 conjugate_gradient(const Matrix0& A, const Vector0& b, const Vector0* x0 = nullptr,
                                  double tol = 1e-5, int max_iter = 1000) {
    size_t ndim = b.size();
    Vector0 x_vector = x0 ? *x0 : Vector0(ndim);

    Vector0 residual = b - A.dot(x_vector);
    Vector0 director = residual;
    double r_norm_sq = residual.dot(residual);

    for (int i = 0; i < max_iter; ++i) {
        Vector0 Ap = A.dot(director);
        double alpha = r_norm_sq / director.dot(Ap);
        x_vector += alpha * director;
        residual -= alpha * Ap;
        double r_norm_sq_new = residual.dot(residual);

        if (std::sqrt(r_norm_sq_new) < tol) {
            return x_vector;
        }

        double beta = r_norm_sq_new / r_norm_sq;
        director = residual + beta * director;
        r_norm_sq = r_norm_sq_new;
    }

    throw std::runtime_error("Conjugate Gradient did not converge after " + std::to_string(max_iter)
                             + " iterations");
}

#endif  // CONJUGATE_GRADIENT_HPP
