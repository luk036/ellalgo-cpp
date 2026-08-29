/**
 * @file lmi_oracle.hpp
 * @brief Oracle for Linear Matrix Inequality feasibility (lazy matrix form)
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
 * @note Concrete oracle in the Template Method pattern: supplies a lazy
 *       `getA` accessor (B - Σ F_k x_k) with a positive `sym_quad` sign to
 *       the shared LmiOracleBase::assess_impl skeleton. Interchangeable with
 *       Lmi0Oracle, LmiOldOracle, ProfitOracle, LowpassOracle, NetworkOracle.
 */
template <typename Arr036, typename Mat = Arr036> class LmiOracle : public LmiOracleBase<Arr036, Mat> {
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
    LmiOracle(size_t ndim, const std::vector<Mat>& F, Mat B)
        : _mgr{ndim}, m_F{F}, m_F0{std::move(B)} {}

    /**
     * @brief
     *
     * @param[in] x
     * @return Cut*
     */
    auto assess_feas(const Arr036& x) -> Cut* {
        const auto n = x.size();

        auto getA = [&n, &x, this](size_t i, size_t j) -> double {
            auto a = this->m_F0(i, j);
            for (auto k = 0U; k != n; ++k) {
                a -= this->m_F[k](i, j) * x[k];
            }
            return a;
        };

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
