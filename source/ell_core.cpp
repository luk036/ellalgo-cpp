#include <cmath>                   // for sqrt
#include <ellalgo/ell_assert.hpp>  // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>    // for EllCalc, EllCalc::Arr
#include <ellalgo/ell_config.hpp>  // for CutStatus, CutStatus::Success
#include <ellalgo/ell_core.hpp>    // for EllCore, EllCore::Arr
#include <tuple>                   // for tuple

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        grad' * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in, out] grad in: gradient; out: xc
 * @return std::tuple<int, double>
 */
using Vec = std::valarray<double>;
