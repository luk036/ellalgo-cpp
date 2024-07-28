// conjugate_gradient.hpp

#ifndef CONJUGATE_GRADIENT2_HPP
#define CONJUGATE_GRADIENT2_HPP

#include <cmath>
#include <stdexcept>
#include <string>

/**
 * Solves a system of linear equations Ax = b using the Conjugate Gradient method.
 *
 * @tparam Matrix The type of the matrix A.
 * @tparam Vector The type of the vectors b and x.
 * @param A The matrix A in the system of linear equations Ax = b.
 * @param b The right-hand side vector b in the system of linear equations Ax = b.
 * @param x_vector The initial guess for the solution vector x. This vector will be updated
 * in-place.
 * @param tol The tolerance for the residual norm, used as the stopping criterion.
 * @param max_iter The maximum number of iterations to perform.
 * @return The solution vector x.
 * @throws std::runtime_error if the Conjugate Gradient method does not converge after the maximum
 * number of iterations.
 */
template <typename Matrix, typename Vector>
Vector conjugate_gradient2(const Matrix& A, const Vector& b, Vector& x_vector, double tol = 1e-5,
                           int max_iter = 1000) {
    using T = typename Vector::value_type;

    Vector residual = b - A * x_vector;
    Vector director = residual;
    T r_norm_sq = residual.dot(residual);

    for (int i = 0; i < max_iter; ++i) {
        Vector Ap = A * director;
        T alpha = r_norm_sq / director.dot(Ap);
        x_vector += alpha * director;
        residual -= alpha * Ap;
        T r_norm_sq_new = residual.dot(residual);

        if (std::sqrt(r_norm_sq_new) < tol) {
            return x_vector;
        }

        T beta = r_norm_sq_new / r_norm_sq;
        director = residual + beta * director;
        r_norm_sq = r_norm_sq_new;
    }

    throw std::runtime_error("Conjugate Gradient did not converge after " + std::to_string(max_iter)
                             + " iterations");
}

#endif  // CONJUGATE_GRADIENT_HPP
