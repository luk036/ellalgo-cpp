/**
 * @file lmi_factory.hpp
 * @brief Factory functions for the LMI oracle family
 */

#pragma once

#include <utility>  // for move
#include <vector>

#include "lmi0_oracle.hpp"
#include "lmi_old_oracle.hpp"
#include "lmi_oracle.hpp"

/**
 * @brief Create an LmiOracle (lazy matrix form)
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 * @param[in] ndim Dimension of the decision space
 * @param[in] F    Vector of matrices F_i (must outlive the oracle)
 * @param[in] B    Constant term (moved in)
 * @return LmiOracle<Arr036, Mat>
 */
template <typename Arr036, typename Mat = Arr036>
inline auto make_lmi_oracle(size_t ndim, const std::vector<Mat>& F, Mat B)
    -> LmiOracle<Arr036, Mat> {
    return {ndim, F, std::move(B)};
}

/**
 * @brief Create an Lmi0Oracle (compact form, no constant term)
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 * @param[in] ndim Dimension of the decision space
 * @param[in] F    Vector of matrices F_i (must outlive the oracle)
 * @return Lmi0Oracle<Arr036, Mat>
 */
template <typename Arr036, typename Mat = Arr036>
inline auto make_lmi0_oracle(size_t ndim, const std::vector<Mat>& F) -> Lmi0Oracle<Arr036, Mat> {
    return {ndim, F};
}

/**
 * @brief Create an LmiOldOracle (explicit matrix form)
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 * @param[in] ndim Dimension of the decision space
 * @param[in] F    Vector of matrices F_i (must outlive the oracle)
 * @param[in] B    Constant term (moved in)
 * @return LmiOldOracle<Arr036, Mat>
 */
template <typename Arr036, typename Mat = Arr036>
inline auto make_lmi_old_oracle(size_t ndim, const std::vector<Mat>& F, Mat B)
    -> LmiOldOracle<Arr036, Mat> {
    return {ndim, F, std::move(B)};
}
