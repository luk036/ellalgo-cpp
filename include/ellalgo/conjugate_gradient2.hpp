// conjugate_gradient.hpp

#ifndef CONJUGATE_GRADIENT2_HPP
#define CONJUGATE_GRADIENT2_HPP

#include <cmath>
#include <stdexcept>

template <typename Matrix, typename Vector>
Vector conjugate_gradient2(const Matrix& A, const Vector& b, const Vector* x0, double tol = 1e-5,
                           int max_iter = 1000) {
    using T = typename Vector::value_type;

    size_t n = b.size();
    Vector x = x0 ? *x0 : Vector(n);

    Vector r = b - A * x;
    Vector p = r;
    T r_norm_sq = r.dot(r);

    for (int i = 0; i < max_iter; ++i) {
        Vector Ap = A * p;
        T alpha = r_norm_sq / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        T r_norm_sq_new = r.dot(r);

        if (std::sqrt(r_norm_sq_new) < tol) {
            return x;
        }

        T beta = r_norm_sq_new / r_norm_sq;
        p = r + beta * p;
        r_norm_sq = r_norm_sq_new;
    }

    throw std::runtime_error("Conjugate Gradient did not converge after " + std::to_string(max_iter)
                             + " iterations");
}

#endif  // CONJUGATE_GRADIENT_HPP
