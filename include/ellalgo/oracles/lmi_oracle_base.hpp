/**
 * @file lmi_oracle_base.hpp
 * @brief Shared LMI oracle skeleton (Template Method)
 */

#pragma once

#include <memory>  // for unique_ptr
#include <utility> // for pair, forward
#include <vector>

#include "ldlt_mgr.hpp"

/**
 * @brief Shared skeleton for Linear Matrix Inequality oracles.
 *
 * The three LMI oracle flavors (LmiOracle, Lmi0Oracle, LmiOldOracle) differ
 * only in how the matrix elements `A(i,j)` are assembled (lazy lambda vs.
 * eagerly-built matrix), in the `sym_quad` sign convention, and in the
 * presence of a constant term `B`. The factorization / witness / cut-packing
 * pipeline is identical and lives here.
 *
 * @note Template Method pattern: `assess_impl` is the fixed algorithm
 *       skeleton (factor -> witness -> sym_quad -> pack cut); each concrete
 *       oracle supplies its own `getA` callable, `sign`, and manager via the
 *       parameter list, keeping every public constructor and member intact.
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 */
template <typename Arr036, typename Mat = Arr036> class LmiOracleBase {
  protected:
    using Cut = std::pair<Arr036, double>;

    /// @brief Storage for the cut returned by assess_feas
    std::unique_ptr<Cut> cut = std::make_unique<Cut>();

    /**
     * @brief Shared assess_feas skeleton: factor, witness, sym_quad, pack cut.
     *
     * @tparam LDLT LDL^T manager type (LDLTMgr)
     * @tparam Fn   Callable with signature double(size_t, size_t)
     * @param[in,out] mgr  LDL^T factorization manager
     * @param[in]     F    Vector of matrices F_i (fixed problem data)
     * @param[in]     sign +1 or -1 convention for sym_quad
     * @param[in]     x    Evaluation point
     * @param[in]     getA Lazy accessor for matrix element A(i, j)
     * @return Cut* pointer to the packed cut, or nullptr if feasible
     */
    template <typename LDLT, typename Fn>
    auto assess_impl(LDLT& mgr, const std::vector<Mat>& F, const int sign, const Arr036& x,
                     Fn&& getA) -> Cut* {
        const auto n = x.size();
        if (mgr.factor(std::forward<Fn>(getA))) {
            return nullptr;
        }
        const auto ep = mgr.witness();  // call before sym_quad() !!!
        Arr036 g{x};
        for (auto i = 0U; i != n; ++i) {
            g[i] = sign * mgr.sym_quad(F[i]);
        }
        this->cut->first = std::move(g);
        this->cut->second = std::move(ep);
        return this->cut.get();
    }
};
