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
template <typename Arr036, typename Mat = Arr036> class LmiOldOracle {
    using Cut = std::pair<Arr036, double>;

  private:
    LDLTMgr _mq;
    const std::vector<Mat> &_F;
    const Mat _F0;
    std::unique_ptr<Cut> cut = std::make_unique<Cut>();

  public:
    /**
     * @brief Construct a new lmi oracle object
     *
     * @param[in] F
     * @param[in] B
     */
    LmiOldOracle(size_t ndim, const std::vector<Mat> &F, Mat B)
        : _mq{ndim}, _F{F}, _F0{std::move(B)} {}
    /**
     * @brief
     *
     * @param[in] x
     * @return std::optional<Cut>
     */
    auto assess_feas(const Arr036 &x) -> Cut * {
        const auto n = x.size();

        Mat A{this->_F0};
        for (auto k = 0U; k != n; ++k) {
            for (auto i = 0U; i != this->_mq._n; ++i) {
                for (auto j = 0U; j != this->_mq._n; ++j) {
                    A(i, j) -= this->_F[k](i, j) * x[k];
                }
            }
        }

        if (this->_mq.factorize(A)) {
            return nullptr;
        }

        auto ep = this->_mq.witness();  // call before sym_quad() !!!
        Arr036 g{x};
        for (auto i = 0U; i != n; ++i) {
            g[i] = this->_mq.sym_quad(this->_F[i]);
        }
        this->cut->first = std::move(g);
        this->cut->second = std::move(ep);
        return this->cut.get();
    }

    /**
     * @brief
     *
     * @param[in] x
     * @return std::optional<Cut>
     */
    auto operator()(const Arr036 &x) -> Cut * { return assess_feas(x); }
};
