#include <ellalgo/ell1d.hpp>             // for ell1d, ell1d::return_t
#include <ellalgo/ell_assert.hpp>        // for ELL_UNLIKELY
#include <ellalgo/ell_config.hpp>        // for CutStatus, CutStatus::Success
#include <ellalgo/half_nonnegative.hpp>  // for half_nonnegative
#include <tuple>                         // for get, tuple

inline double my_abs(const double &a) { return a > 0.0 ? a : -a; }

/**
 * @brief
 *
 * @param[in] cut
 * @return ell1d::return_t
 */
auto ell1d::update(const std::pair<double, double> &cut) noexcept -> CutStatus {
    // const auto& [g, beta] = cut;
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
 * @brief
 *
 * @param[in] cut
 * @return ell1d::return_t
 */
auto ell1d::update_cc(const std::pair<double, double> &cut) noexcept -> CutStatus {
    const auto &g = cut.first;
    const auto tau = ::my_abs(this->_r * g);
    this->_tsq = tau * tau;
    this->_r /= 2;
    this->_xc += g > 0.0 ? -this->_r : this->_r;
    return CutStatus::Success;
}
