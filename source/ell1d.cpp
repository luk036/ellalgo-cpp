#include <ellalgo/cut_config.hpp>        // for CUTStatus, CUTStatus::success
#include <ellalgo/ell1d.hpp>             // for ell1d, ell1d::return_t
#include <ellalgo/ell_assert.hpp>        // for ELL_UNLIKELY
#include <ellalgo/half_nonnegative.hpp>  // for half_nonnegative
#include <tuple>                         // for get, tuple

inline double my_abs(const double& a) { return a > 0.0 ? a : -a; }

/**
 * @brief
 *
 * @param[in] cut
 * @return ell1d::return_t
 */
auto ell1d::update(const std::tuple<double, double>& cut) noexcept -> ell1d::return_t {
    // const auto& [g, beta] = cut;
    const auto& g = std::get<0>(cut);
    const auto& beta = std::get<1>(cut);

    const auto tau = ::my_abs(this->_r * g);
    const auto tsq = tau * tau;

    if (beta == 0.) {
        this->_r /= 2;
        this->_xc += g > 0. ? -this->_r : this->_r;
        return {CUTStatus::success, tsq};
    }
    if (beta > tau) {
        return {CUTStatus::nosoln, tsq};  // no sol'n
    }
    if (ELL_UNLIKELY(beta < -tau)) {
        return {CUTStatus::noeffect, tsq};  // no effect
    }

    const auto bound = this->_xc - beta / g;
    const auto u = g > 0. ? bound : this->_xc + this->_r;
    const auto l = g > 0. ? this->_xc - this->_r : bound;

    this->_r = algo::half_nonnegative(u - l);
    this->_xc = l + this->_r;
    return {CUTStatus::success, tsq};
}
