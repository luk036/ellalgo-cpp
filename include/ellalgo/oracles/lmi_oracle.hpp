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
 *        s.t.  (B - F * x) >= 0
 */
template <typename Arr036, typename Mat = Arr036> class LmiOracle {
  using Cut = std::pair<Arr036, double>;

private:
  LDLTMgr _Q;
  const std::vector<Mat> &_F;
  const Mat _F0;
  std::unique_ptr<Cut> cut;

public:
  /**
   * @brief Construct a new lmi oracle object
   *
   * @param[in] F
   * @param[in] B
   */
  LmiOracle(size_t dim, const std::vector<Mat> &F, Mat B)
      : _Q{dim}, _F{F}, _F0{std::move(B)}, cut{std::unique_ptr<Cut>(
                                               new Cut{})} {}

  /**
   * @brief
   *
   * @param[in] x
   * @return std::optional<Cut>
   */
  auto assess_feas(const Arr036 &x) -> Cut * {
    const auto n = x.size();

    auto getA = [&, this](size_t i, size_t j) -> double {
      auto a = this->_F0(i, j);
      for (auto k = 0U; k != n; ++k) {
        a -= this->_F[k](i, j) * x[k];
      }
      return a;
    };

    if (this->_Q.factor(getA))
      return nullptr;
    auto ep = this->_Q.witness(); // call before sym_quad() !!!
    Arr036 g = x;
    // auto g = Arr036(x.size(), 0.0);
    for (auto i = 0U; i != n; ++i) {
      g[i] = this->_Q.sym_quad(this->_F[i]);
    }
    cut->first = std::move(g);
    cut->second = std::move(ep);
    return cut.get();
  }

  /**
   * @brief
   *
   * @param[in] x
   * @return std::optional<Cut>
   */
  auto operator()(const Arr036 &x) -> Cut * { return assess_feas(x); }
};
