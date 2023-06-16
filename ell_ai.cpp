#include <cmath>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

using Mat = std::vector<std::vector<double>>;
using ArrayType = std::vector<double>;
using CutChoice = std::variant<double, ArrayType>;
using Cut = std::tuple<ArrayType, CutChoice>;

/**
 * @brief
 *
 */
enum class CutStatus { Success, Infeasible, Unbounded };

/**
 * @brief
 *
 */
class Ell {
public:
  bool no_defer_trick = false;
  Mat _mq;
  ArrayType _xc;
  double _kappa;
  double _tsq;
  // EllCalc _helper;

  /**
   * @brief Construct a new Ell object
   *
   * @param val
   * @param xc
   */
  Ell(int val, ArrayType xc) {
    int ndim = xc.size();
    // _helper = EllCalc(ndim);
    _xc = xc;
    _tsq = 0.0;
    if (typeid(val) == typeid(int) || typeid(val) == typeid(double)) {
      _kappa = val;
      _mq.resize(ndim, ArrayType(ndim, 0.0));
      for (int i = 0; i < ndim; i++) {
        _mq[i][i] = 1.0;
      }
    } else {
      _kappa = 1.0;
      _mq.resize(ndim, ArrayType(ndim, 0.0));
      for (int i = 0; i < ndim; i++) {
        _mq[i][i] = val[i];
      }
    }
  }

  /**
   * @brief
   *
   * @param cut
   * @param cut_strategy
   * @return CutStatus
   */
  CutStatus
  _update_core(Cut cut,
               std::function<CutStatus(CutChoice, double)> cut_strategy) {
    ArrayType grad = std::get<0>(cut);
    CutChoice beta = std::get<1>(cut);
    ArrayType grad_t;
    for (int i = 0; i < _mq.size(); i++) {
      double sum = 0.0;
      for (int j = 0; j < _mq.size(); j++) {
        sum += _mq[i][j] * grad[j];
      }
      grad_t.push_back(sum);
    }
    double omega = 0.0;
    for (int i = 0; i < grad.size(); i++) {
      omega += grad[i] * grad_t[i];
    }
    _tsq = _kappa * omega;
    CutStatus status;
    double rho, sigma, delta;
    std::tie(status, rho, sigma, delta) = cut_strategy(beta, _tsq);
    if (status != CutStatus::Success) {
      return status;
    }
    for (int i = 0; i < _mq.size(); i++) {
      _xc[i] -= (rho / omega) * grad_t[i];
      for (int j = 0; j < _mq.size(); j++) {
        _mq[i][j] -= (sigma / omega) * grad_t[i] * grad_t[j];
      }
    }
    _kappa *= delta;
    if (no_defer_trick) {
      for (int i = 0; i < _mq.size(); i++) {
        for (int j = 0; j < _mq.size(); j++) {
          _mq[i][j] *= _kappa;
        }
      }
      _kappa = 1.0;
    }
    return status;
  }
};

using SearchSpace = std::vector<std::tuple<ArrayType, ArrayType>>;
using SearchSpaceQ = std::vector<std::tuple<ArrayType, double>>;
