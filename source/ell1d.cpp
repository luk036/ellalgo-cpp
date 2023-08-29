#include <ellalgo/ell1d.hpp>             // for ell1d, ell1d::return_t
#include <ellalgo/ell_assert.hpp>        // for ELL_UNLIKELY
#include <ellalgo/ell_config.hpp>        // for CutStatus, CutStatus::Success
#include <ellalgo/half_nonnegative.hpp>  // for half_nonnegative
#include <tuple>                         // for get, tuple

/**
 * The function "my_abs" returns the absolute value of a given double number.
 *
 * @param[in] a a is a constant reference to a double value.
 *
 * @return The function `my_abs` returns the absolute value of the input `a`.
 */
inline double my_abs(const double &number) { return number > 0.0 ? number : -number; }

/**
 * The function updates the parameters of an ellipsoidal cut based on a given cut.
 *
 * @param[in] cut The `cut` parameter is a `std::pair<double, double>` representing a cut. The first
 * element of the pair, `cut.first`, is the value of `g`, and the second element, `cut.second`, is
 * the value of `beta`.
 *
 * @return The function `update` returns a value of type `CutStatus`.
 */
auto ell1d::update(const std::pair<double, double> &cut) noexcept -> CutStatus {
    const auto &g = cut.first;
    const auto &beta = cut.second;

    const auto tau = ::my_abs(this->_r * g);
    this->_tsq = tau * tau;

    // if (beta == 0.0) {
    //   this->_r /= 2;
    //   this->_xc += g > 0.0 ? -this->_r : this->_r;
    //   return CutStatus::Success;
    // }
    if (beta > tau) {
        return CutStatus::NoSoln;  // no sol'n
    }
    if (ELL_UNLIKELY(beta < -tau)) {
        return CutStatus::NoEffect;  // no effect
    }

    const auto bound = this->_xc - beta / g;
    const auto u = g > 0.0 ? bound : this->_xc + this->_r;
    const auto l = g > 0.0 ? this->_xc - this->_r : bound;

    this->_r = algo::half_nonnegative(u - l);
    this->_xc = l + this->_r;
    return CutStatus::Success;
}

/**
 * The function updates the central cut of an ellipsoid based on a given cut.
 *
 * @param[in] cut The parameter `cut` is a `std::pair<double, double>` which represents a cut in the
 * form of a pair of values. The first value `cut.first` represents the gradient `g` of the cut, and
 * the second value `cut.second` is not used in this function.
 *
 * @return a value of type `CutStatus`.
 */
auto ell1d::update_central_cut(const std::pair<double, double> &cut) noexcept -> CutStatus {
    const auto &g = cut.first;
    const auto tau = ::my_abs(this->_r * g);
    this->_tsq = tau * tau;
    this->_r /= 2;
    this->_xc += g > 0.0 ? -this->_r : this->_r;
    return CutStatus::Success;
}
