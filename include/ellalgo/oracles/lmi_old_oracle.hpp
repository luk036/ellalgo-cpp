/**
 * @file lmi_old_oracle.hpp
 * @brief Oracle for Linear Matrix Inequality feasibility (explicit matrix form)
 */

// -*- coding: utf-8 -*-
#pragma once

#include <utility>  // for move
#include <vector>

#include "lmi_oracle_base.hpp"

/**
 * @brief Oracle for Linear Matrix Inequality.
 *
 *    This oracle solves the following feasibility problem:
 *
 *        find  x
 *        s.t.  (B - F * x) >= 0
 *
 * @note Concrete oracle in the Template Method pattern: builds the matrix
 *       `A = B - Σ F_k x_k` eagerly and feeds a lambda accessor with a
 *       positive `sym_quad` sign to the shared LmiOracleBase::assess_impl
 *       skeleton. Behaviorally identical to LmiOracle, which uses lazy
 *       evaluation instead of a pre-built matrix.
 */
template <typename Arr036, typename Mat = Arr036>
class LmiOldOracle : public LmiOracleBase<Arr036, Mat> {
    using Base = LmiOracleBase<Arr036, Mat>;
    using Cut = std::pair<Arr036, double>;

    LDLTMgr _mgr;
    const std::vector<Mat>& m_F;
    Mat m_F0;

  public:
    /**
     * @brief Construct a new lmi oracle object
     *
     * @param[in] ndim
     * @param[in] F
     * @param[in] B
     */
    LmiOldOracle(size_t ndim, const std::vector<Mat>& F, Mat B)
        : _mgr{ndim}, m_F{F}, m_F0{std::move(B)} {}
    /**
     * @brief Assess the feasibility of a given point via LDLT factorization
     *
     * @param[in] x The point to assess feasibility
     * @return Cut* Pointer to cut information, or nullptr if feasible
     */
    auto assess_feas(const Arr036& x) -> Cut* {
        const auto n = x.size();

        Mat A{this->m_F0};
        for (auto k = 0U; k != n; ++k) {
            for (auto i = 0U; i != this->_mgr._n; ++i) {
                for (auto j = 0U; j != this->_mgr._n; ++j) {
                    A(i, j) -= this->m_F[k](i, j) * x[k];
                }
            }
        }

        auto getA = [&A](size_t i, size_t j) { return A(i, j); };
        return this->assess_impl(this->_mgr, this->m_F, +1, x, getA);
    }

    /**
     * @brief Call operator wrapping assess_feas
     *
     * @param[in] x The point to assess feasibility
     * @return Cut* Pointer to cut, or nullptr if feasible
     */
    auto operator()(const Arr036& x) -> Cut* { return assess_feas(x); }
};
