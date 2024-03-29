// -*- coding: utf-8 -*-
#pragma once

#include <memory>  // for unique_ptr
#include <vector>

#include "ldlt_mgr.hpp"

/**
 * @brief Oracle for Linear Matrix Inequality.
 *
 *    This oracle solves the following feasibility problem:
 *
 *        find  x
 *        s.t.  (B - F * x) >= 0
 */
template <typename Arr036, typename Mat = Arr036> class LmiOracle {
    using Cut = std::pair<Arr036, double>;

    LDLTMgr _mgr;
    const std::vector<Mat> &_F;
    const Mat _F0;
    std::unique_ptr<Cut> cut = std::make_unique<Cut>();

  public:
    /**
     * @brief Construct a new lmi oracle object
     *
     * @param[in] ndim
     * @param[in] F
     * @param[in] B
     */
    LmiOracle(size_t ndim, const std::vector<Mat> &F, Mat B)
        : _mgr{ndim}, _F{F}, _F0{std::move(B)} {}

    /**
     * @brief
     *
     * @param[in] x
     * @return Cut*
     */
    auto assess_feas(const Arr036 &x) -> Cut * {
        const auto n = x.size();

        auto getA = [&n, &x, this](size_t i, size_t j) -> double {
            auto a = this->_F0(i, j);
            for (auto k = 0U; k != n; ++k) {
                a -= this->_F[k](i, j) * x[k];
            }
            return a;
        };

        if (this->_mgr.factor(getA)) {
            return nullptr;
        }

        auto ep = this->_mgr.witness();  // call before sym_quad() !!!
        Arr036 g{x};
        for (auto i = 0U; i != n; ++i) {
            g[i] = this->_mgr.sym_quad(this->_F[i]);
        }
        this->cut->first = std::move(g);
        this->cut->second = std::move(ep);
        return this->cut.get();
    }

    /**
     * @brief
     *
     * @param[in] x
     * @return Cut*
     */
    auto operator()(const Arr036 &x) -> Cut * { return assess_feas(x); }
};
