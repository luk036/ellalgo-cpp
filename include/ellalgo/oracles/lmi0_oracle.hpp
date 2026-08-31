/**
 * @file lmi0_oracle.hpp
 * @brief Oracle for Linear Matrix Inequality (LMI) feasibility (compact form)
 */

// -*- coding: utf-8 -*-
#pragma once

#include <utility>  // for move
#include <vector>

#include "lmi_oracle_base.hpp"

/**
 * @brief Oracle for Linear Matrix Inequality (LMI) feasibility problems
 *
 * This oracle solves the following feasibility problem:
 *
 *     find  x
 *     s.t.  F₀ + F₁x₁ + F₂x₂ + ... + Fₙxₙ ⪯ 0
 *
 * where Fᵢ are symmetric matrices and ⪯ denotes negative semidefinite.
 * This is a fundamental problem in semidefinite programming and control theory.
 *
 * @note Concrete oracle in the Template Method pattern: supplies a lazy
 *       `getA` accessor (Σ F_k x_k) with a negative `sym_quad` sign to the
 *       shared LmiOracleBase::assess_impl skeleton.
 *
 * @tparam Arr036 Array type for the decision variables (size 3-6)
 * @tparam Mat Matrix type (defaults to Arr036)
 */
template <typename Arr036, typename Mat = Arr036> class Lmi0Oracle
    : public LmiOracleBase<Arr036, Mat> {
    using Base = LmiOracleBase<Arr036, Mat>;
    using Cut = std::pair<Arr036, double>;

  public:
    LDLTMgr _mq;  ///< LDLT manager for matrix factorization

  private:
    const std::vector<Mat>& m_F;  ///< Vector of matrices F₀, F₁, ..., Fₙ

  public:
    /**
     * @brief Construct a new LMI Oracle object
     *
     * @param[in] ndim Dimension of the decision space
     * @param[in] F Vector of matrices defining the LMI constraints
     */
    Lmi0Oracle(size_t ndim, const std::vector<Mat>& F) : _mq(ndim), m_F{F} {}

    /**
     * @brief Assess the feasibility of a given point
     *
     * This method checks if the given point x satisfies the LMI constraint.
     * If not feasible, it returns a cutting plane that separates x from
     * the feasible region.
     *
     * @param[in] x The point to assess feasibility
     * @return Pointer to cut information, or nullptr if feasible
     */
    auto assess_feas(const Arr036& x) -> Cut* {
        const auto n = x.size();

        auto getA = [&n, &x, this](size_t i, size_t j) -> double {
            auto a = 0.0;
            for (auto k = 0U; k != n; ++k) {
                a += this->m_F[k](i, j) * x[k];
            }
            return a;
        };

        return this->assess_impl(this->_mq, this->m_F, -1, x, getA);
    }

    /**
     * @brief Call operator wrapping assess_feas
     *
     * @param[in] x The point to assess feasibility
     * @return Cut* Pointer to cut, or nullptr if feasible
     */
    auto operator()(const Arr036& x) -> Cut* { return assess_feas(x); }
};
