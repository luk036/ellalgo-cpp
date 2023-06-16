// -*- coding: utf-8 -*-
#pragma once

#include "ldlt_mgr.hpp"
#include <memory> // for unique_ptr
#include <vector>

/**
 * @brief Oracle for Linear Matrix Inequality.
 *
 *    This oracle solves the following feasibility problem:
 *
 *        find  x
 *        s.t.  F * x <= 0
 */
template <typename Arr036, typename Mat = Arr036> class Lmi0Oracle {
  using Cut = std::pair<Arr036, double>;

public:
  LDLTMgr _mq;

private:
  const std::vector<Mat> &_F;
  std::unique_ptr<Cut> cut;

public:
  /**
   * @brief Construct a new Lmi 0 Oracle object
   *
   * @param[in] ndim
   * @param[in] F
   */
  Lmi0Oracle(size_t ndim, const std::vector<Mat> &F)
      : _mq(ndim), _F{F}, cut{std::unique_ptr<Cut>(new Cut{})} {}

  /**
   * @brief
   *
   * @param[in] x
   * @return Cut*
   */
  auto assess_feas(const Arr036 &x) -> Cut * {
    const auto n = x.size();

    auto getA = [&, this](size_t i, size_t j) -> double {
      auto a = 0.0;
      for (auto k = 0U; k != n; ++k) {
        a += this->_F[k](i, j) * x[k];
      }
      return a;
    };

    if (this->_mq.factor(getA)) {
      return nullptr;
    }

    auto ep = this->_mq.witness(); // call before sym_quad() !!!
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
  auto operator()(const Arr036 &x) -> Cut * { return assess_feas(x); }
};
