// -*- coding: utf-8 -*-
#pragma once

#include <cassert>  // for assert
#include <cstddef>  // for size_t
#include <functional>
#include <utility>  // for pair
#include <valarray>

#include "../ell_matrix.hpp"

/**
 * @brief LDLT factorization
 *
 * `LDLTMgr` is a class that performs the LDLT factorization for a given
 * symmetric matrix. The LDLT factorization decomposes a symmetric matrix A into
 * the product of a lower triangular matrix L, a diagonal matrix D, and the
 * transpose of L. This factorization is useful for solving linear systems and
 * eigenvalue problems. The class provides methods to perform the factorization,
 * check if the matrix is positive definite, calculate a witness vector if it is
 * not positive definite, and calculate the symmetric quadratic form.
 *
 *  - LDL^T square-root-free version
 *  - Option allow semidefinite
 *  - A matrix A in R^{m x m} is positive definite iff v' A v > 0
 *      for all v in R^n.
 *  - O(pos^2) per iteration, independent of N
 */
class LDLTMgr {
    using Vec = std::valarray<double>;
    using Rng = std::pair<size_t, size_t>;

  public:
    Rng pos{0U, 0U};  //!< the rows where the process starts and stops
    Vec witness_vec;  //!< witness vector
    const size_t _n;  //!< dimension

  private:
    Matrix T;  //!< temporary storage

  public:
    /**
     * @brief Construct a new ldlt ext object
     *
     * @param[in] N dimension
     */
    explicit LDLTMgr(size_t N) : witness_vec(0.0, N), _n{N}, T{N} {}

    LDLTMgr(const LDLTMgr&) = delete;
    LDLTMgr& operator=(const LDLTMgr&) = delete;
    LDLTMgr(LDLTMgr&&) = default;

    /**
     * @brief Perform LDLT Factorization
     *
     * The `factorize` function is a template function that takes a symmetric
     * matrix `A` as input and performs the LDLT factorization on it. It calls
     * the `factor` function with a lambda function as an argument. The lambda
     * function takes the indices `i` and `j` and returns the element `A(i, j)`
     * of the matrix `A`. The `factor` function performs the actual
     * factorization using the provided lambda function. The `factorize`
     * function returns a boolean value indicating whether the factorization was
     * successful or not.
     *
     * @param[in] A Symmetric Matrix
     */
    template <typename Mat> auto factorize(const Mat& A) -> bool {
        return this->factor([&A](size_t i, size_t j) { return A(i, j); });
    }

    /**
     * @brief Perform LDLT Factorization (Lazy evaluation)
     *
     * @param[in] get_matrix_elem function to access the elements of A
     * @return true
     * @return false
     *
     * See also: factorize()
     */
    auto factor(const std::function<double(size_t, size_t)>& get_matrix_elem) -> bool;

    /**
     * @brief Perform LDLT Factorization (Lazy evaluation)
     *
     * @param[in] get_matrix_elem function to access the elements of A
     * @return true
     * @return false
     *
     * See also: factorize()
     */
    auto factor_with_allow_semidefinite(
        const std::function<double(size_t, size_t)>& get_matrix_elem) -> bool;

    /**
     * @brief Check if the matrix is symmetric positive definite.
     *
     * @return bool True if the matrix is SPD, false otherwise.
     */
    constexpr auto is_spd() const noexcept -> bool { return this->pos.second == 0; }

    /**
     * @brief witness that certifies $A$ is not symmetric positive definite
     * (spd)
     *
     * The `witness()` function calculates a witness that certifies that the
     * matrix `A` is not symmetric positive definite (spd). It returns a
     * `double` value that represents the witness.
     *
     * @return double
     */
    auto witness() -> double;

    /**
     * @brief Set the witness vec object
     *
     * @tparam Arr036
     * @param[in] v
     */
    template <typename Arr036> auto set_witness_vec(Arr036& v) const -> void {
        for (auto i = 0U; i != this->_n; ++i) {
            v[i] = this->witness_vec[i];
        }
    }

    /**
     * @brief Calculate v'*{A}(pos,pos)*v
     *
     * @tparam Mat
     * @param[in] A
     * @return double
     */
    template <typename Mat> auto sym_quad(const Mat& A) const -> double {
        auto res = double{};
        const auto& v = this->witness_vec;
        // const auto& [start, stop] = this->pos;
        const auto& start = this->pos.first;
        const auto& stop = this->pos.second;
        for (auto i = start; i != stop; ++i) {
            auto s = 0.0;
            for (auto j = i + 1; j != stop; ++j) {
                s += A(i, j) * v[j];
            }
            res += v[i] * (A(i, i) * v[i] + 2.0 * s);
        }
        return res;
    }

    /**
     * @brief Return upper triangular matrix $R$ where $A = R^T R$
     *
     * The `sqrt` function calculates the square root of a symmetric positive
     * definite matrix `M`. It assumes that `M` is a zero matrix and calculates
     * the upper triangular matrix `R` such that `M = R^T * R`.
     *
     * @tparam Mat
     * @param[in,out] M
     */
    template <typename Mat> auto sqrt(Mat& M) -> void {
        assert(this->is_spd());

        for (auto i = 0U; i != this->_n; ++i) {
            M(i, i) = std::sqrt(this->T(i, i));
            for (auto j = i + 1; j != this->_n; ++j) {
                M(i, j) = this->T(j, i) * M(i, i);
                M(j, i) = 0.0;
            }
        }
    }
};
