/**
 * @file ell_stable.hpp
 * @brief Numerically stable ellipsoid search space (LDL^T strategy)
 */

#pragma once

#include <valarray>

#include "ell_base.hpp"

/**
 * @brief Ellipsoid Search Space (stable strategy)
 *
 * Concrete strategy of `EllBase` with the numerically-stable LDL^T update:
 * `EllCore::update_stable_*` is used for cutting-plane updates.
 *
 * @f[
 *     \mathcal{E} = \{x \mid (x - x_c)^T Q^{-1} (x - x_c) \le \kappa\}
 * @f]
 *
 * @tparam Arr Array type of the center point
 */
template <typename Arr> class EllStable : public EllBase<Arr, true> {
    using Base = EllBase<Arr, true>;

  public:
    using Vec = std::valarray<double>;
    using ArrayType = Arr;

    /**
     * @brief Named constructor: initial ellipsoid from per-axis radii.
     *
     * @param[in] val A vector of per-axis radii (diagonal of the shape matrix).
     * @param[in] x An array of type Arr. This parameter is moved.
     * @return EllStable A new EllStable object.
     */
    static auto from_radii(const Vec& val, Arr x) -> EllStable {
        return EllStable(val, std::move(x));
    }

    /**
     * @brief Named constructor: initial ellipsoid from a scaling factor.
     *
     * @param[in] alpha A double value representing the scaling factor.
     * @param[in] x An array of type Arr. This parameter is moved.
     * @return EllStable A new EllStable object.
     */
    static auto from_alpha(const double alpha, Arr x) -> EllStable {
        return EllStable(alpha, std::move(x));
    }

    /// @brief Construct from a diagonal vector and a center point (moved in).
    EllStable(const Vec& val, Arr x) : Base(val, std::move(x)) {}

    /// @brief Construct from a scaling factor and a center point (moved in).
    EllStable(const double alpha, Arr x) : Base(alpha, std::move(x)) {}

    /// @brief Move constructor.
    EllStable(EllStable&& E) noexcept = default;

    /// @brief Move assignment operator.
    auto operator=(EllStable&&) noexcept -> EllStable& = default;

    /// @brief Destructor.
    ~EllStable() = default;

    /// @brief Explicit copy constructor.
    explicit EllStable(const EllStable& E) = default;

    /// @brief Deleted copy assignment operator (non-copyable).
    auto operator=(const EllStable& E) -> EllStable& = delete;

    /**
     * @brief Explicitly copy the EllStable object.
     *
     * @return EllStable A new EllStable object that is a copy of the current object.
     */
    auto copy() const -> EllStable { return EllStable(*this); }
};  // } EllStable
