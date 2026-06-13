/**
 * @file ldlt_mgr.cpp
 * @brief Implementation of LDLT factorization witness computation
 *
 * Computes a witness vector certifying that a matrix
 * is not symmetric positive definite.
 */

#include <ellalgo/oracles/ldlt_mgr.hpp>

/**
 * @brief Compute a witness certifying non-SPD status
 *
 * After a failed LDLT factorization (is_spd() == false),
 * this function computes a witness vector w such that
 * w' A w < 0, proving A is not positive definite.
 *
 * @return double The negative pivot value (-T(m,m)) at the failure point
 */
auto LDLTMgr::witness() -> double {
    assert(!this->is_spd());

    const auto& start = this->pos.first;
    const auto& n = this->pos.second;
    auto m = n - 1;
    this->witness_vec[m] = 1.0;
    for (auto i = m; i > start; --i) {
        this->witness_vec[i - 1] = 0.0;
        for (auto k = i; k != n; ++k) {
            this->witness_vec[i - 1] -= this->T(k, i - 1) * this->witness_vec[k];
        }
    }
    return -this->T(m, m);
}
