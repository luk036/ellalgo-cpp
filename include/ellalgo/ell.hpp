/**
 * @file ell.hpp
 * @brief Ellipsoid search space (classic Q-update strategy)
 */

#pragma once

#include <valarray>

#include "ell_base.hpp"

/**
 * @brief Ellipsoid Search Space (classic strategy)
 *
 * Concrete strategy of `EllBase` with the classic direct Q-update:
 * `EllCore::update_*` is used for cutting-plane updates.
 *
 * @f[
 *     \mathcal{E} = \{x \mid (x - x_c)^T Q^{-1} (x - x_c) \le \kappa\}
 * @f]
 *
 * @tparam Arr Array type of the center point
 */
template <typename Arr> class Ell : public EllBase<Arr, false> {
    using Base = EllBase<Arr, false>;

  public:
    using Vec = std::valarray<double>;
    using ArrayType = Arr;

    /**
     * @brief Named constructor: initial ellipsoid from per-axis radii.
     *
     * @param[in] val A vector of per-axis radii (diagonal of the shape matrix).
     * @param[in] x An array of type Arr. This parameter is moved.
     * @return Ell A new Ell object.
     */
    static auto from_radii(const Vec& val, Arr x) -> Ell { return Ell(val, std::move(x)); }

    /**
     * @brief Named constructor: initial ellipsoid from a scaling factor.
     *
     * @param[in] alpha A double value representing the scaling factor.
     * @param[in] x An array of type Arr. This parameter is moved.
     * @return Ell A new Ell object.
     */
    static auto from_alpha(const double alpha, Arr x) -> Ell { return Ell(alpha, std::move(x)); }

    /// @brief Construct from a diagonal vector and a center point (moved in).
    Ell(const Vec& val, Arr x) : Base(val, std::move(x)) {}

    /// @brief Construct from a scaling factor and a center point (moved in).
    Ell(const double alpha, Arr x) : Base(alpha, std::move(x)) {}

    /// @brief Move constructor.
    Ell(Ell&& E) noexcept = default;

    /// @brief Move assignment operator.
    auto operator=(Ell&& E) noexcept -> Ell& = default;

    /// @brief Destructor.
    ~Ell() = default;

    /// @brief Explicit copy constructor.
    explicit Ell(const Ell& E) = default;

    /// @brief Deleted copy assignment operator (non-copyable).
    auto operator=(const Ell& E) -> Ell& = delete;

    /**
     * @brief Explicitly copy the Ell object.
     *
     * @return Ell A new Ell object that is a copy of the current object.
     */
    auto copy() const -> Ell { return Ell(*this); }
};  // } Ell
