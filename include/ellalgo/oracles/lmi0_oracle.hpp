// -*- coding: utf-8 -*-
#pragma once

#include <memory>  // for unique_ptr
#include <vector>

#include "ldlt_mgr.hpp"

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
 * @tparam Arr036 Array type for the decision variables (size 3-6)
 * @tparam Mat Matrix type (defaults to Arr036)
 */
template <typename Arr036, typename Mat = Arr036> class Lmi0Oracle {
    using Cut = std::pair<Arr036, double>;

  public:
    LDLTMgr _mq;  ///< LDLT manager for matrix factorization

  private:
    const std::vector<Mat>& _F;  ///< Vector of matrices F₀, F₁, ..., Fₙ
    std::unique_ptr<Cut> cut = std::make_unique<Cut>();  ///< Storage for cut information

  public:
    /**
     * @brief Construct a new LMI Oracle object
     *
     * @param[in] ndim Dimension of the decision space
     * @param[in] F Vector of matrices defining the LMI constraints
     */
    Lmi0Oracle(size_t ndim, const std::vector<Mat>& F) : _mq(ndim), _F{F} {}

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
                a += this->_F[k](i, j) * x[k];
            }
            return a;
        };

        if (this->_mq.factor(getA)) {
            return nullptr;
        }

        auto ep = this->_mq.witness();  // call before sym_quad() !!!
        Arr036 g{x};
        for (auto i = 0U; i != n; ++i) {
            g[i] = -this->_mq.sym_quad(this->_F[i]);
        }
        cut->first = std::move(g);
        cut->second = std::move(ep);
        return cut.get();
    }

    /**
     * @brief
     *
     * @param[in] x
     * @return Cut*
     */
    auto operator()(const Arr036& x) -> Cut* { return assess_feas(x); }
};
