/**
 * @file ldlt_mgr.cpp
 * @brief Implementation of LDLT factorization manager
 *
 * This file implements the LDLT factorization algorithm for symmetric matrices.
 * The LDLT factorization decomposes a symmetric matrix A into L*D*L^T where
 * L is lower triangular with unit diagonal and D is diagonal.
 */

#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <functional>

/**
 * @brief Perform LDLT factorization of a symmetric matrix
 *
 * This function performs the LDLT factorization of a symmetric matrix using
 * the provided function to access matrix elements. It stores the result in
 * the internal T matrix and updates the position information.
 *
 * @param[in] get_matrix_elem Function to access matrix elements A(i,j)
 * @return true if the matrix is positive definite, false otherwise
 */
auto LDLTMgr::factor(const std::function<double(size_t, size_t)>& get_matrix_elem) -> bool {
    this->pos = {0U, 0U};
    auto const& start = this->pos.first;
    auto& stop = this->pos.second;

    for (auto i = 0U; i != this->_n; ++i) {
        auto d = get_matrix_elem(i, start);
        for (auto j = start; j != i; ++j) {
            this->T(j, i) = d;
            this->T(i, j) = d / this->T(j, j);  // note: T(j, i) here!
            auto s = j + 1;
            d = get_matrix_elem(i, s);
            for (auto k = start; k != s; ++k) {
                d -= this->T(i, k) * this->T(k, s);
            }
        }
        this->T(i, i) = d;

        if (d <= 0.0) {
            stop = i + 1;
            break;
        }
    }

    return this->is_spd();
}

/**
 * @brief Perform LDLT factorization allowing semidefinite matrices
 *
 * This function performs LDLT factorization but allows for semidefinite
 * matrices (zero diagonal elements). When a zero diagonal element is
 * encountered, it restarts the factorization from the next position.
 *
 * @param[in] get_matrix_elem Function to access matrix elements A(i,j)
 * @return true if the matrix is positive definite, false otherwise
 */
auto LDLTMgr::factor_with_allow_semidefinite(
    const std::function<double(size_t, size_t)>& get_matrix_elem) -> bool {
    this->pos = {0U, 0U};
    auto& start = this->pos.first;
    auto& stop = this->pos.second;

    for (auto i = 0U; i != this->_n; ++i) {
        auto d = get_matrix_elem(i, start);
        for (auto j = start; j != i; ++j) {
            this->T(j, i) = d;
            this->T(i, j) = d / this->T(j, j);  // note: T(j, i) here!
            auto s = j + 1;
            d = get_matrix_elem(i, s);
            for (auto k = start; k != s; ++k) {
                d -= this->T(i, k) * this->T(k, s);
            }
        }
        this->T(i, i) = d;

        if (d < 0.0) {
            // this->stop = i + 1;
            stop = i + 1;
            break;
        }
        if (d == 0.0) {
            start = i + 1;
            // restart at i + 1, special as an LMI oracle
        }
    }
    return this->is_spd();
}

/**
 * @brief Calculate witness vector for non-positive definite matrix
 *
 * This function calculates a witness vector that certifies that the matrix
 * is not positive definite. The witness is used in cutting-plane methods
 * to generate separating hyperplanes.
 *
 * @return The negative of the last diagonal element in the factorization
 */
auto LDLTMgr::witness() -> double {
    assert(!this->is_spd());

    // const auto& [start, n] = this->pos;
    const auto& start = this->pos.first;
    const auto& n = this->pos.second;
    auto m = n - 1;  // assume stop > 0
    this->witness_vec[m] = 1.0;
    for (auto i = m; i > start; --i) {
        this->witness_vec[i - 1] = 0.0;
        for (auto k = i; k != n; ++k) {
            this->witness_vec[i - 1] -= this->T(k, i - 1) * this->witness_vec[k];
        }
    }
    return -this->T(m, m);
}
