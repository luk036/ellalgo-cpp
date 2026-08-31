/**
 * @file ell_base.hpp
 * @brief Shared ellipsoid search space base (compile-time Strategy)
 */

#pragma once

#include <cstddef>
#include <utility>
#include <valarray>

#include "ell_config.hpp"
#include "ell_core.hpp"

/**
 * @brief Ellipsoid Search Space (shared base)
 *
 * The `EllBase` class represents an ellipsoid search space:
 *
 * @f[
 *     \mathcal{E} = \{x \mid (x - x_c)^T Q^{-1} (x - x_c) \le \kappa\}
 * @f]
 *
 * This version keeps $Q$ symmetric but no promise of positive definite.
 *
 * @note Strategy pattern: the non-type template parameter `Stable` selects
 *       the ellipsoid-update strategy at compile time. When `Stable == true`
 *       the numerically-stable LDL^T update path is used
 *       (EllCore::update_stable_*); otherwise the classic direct Q-update is
 *       used (EllCore::update_*). `Ell` and `EllStable` are thin subclasses
 *       pinning this strategy while exposing identical public APIs.
 *
 * <pre>
 *    n-dimensional space
 *         ┌─┐
 *       ┌─┘ └─┐
 *     ┌─┘     └─┐
 *   ┌─┘         └─┐  ←─ ellipsoid boundary
 *   │   ● xc      │      center point
 *   └─┐         ┌─┘
 *     └─┐     ┌─┘
 *       └─┐ ┌─┘
 *         └─┘
 * </pre>
 *
 * @tparam Arr    Array type of the center point
 * @tparam Stable Compile-time strategy selector (true = LDL^T stable updates)
 */
template <typename Arr, bool Stable> class EllBase {
  public:
    using Vec = std::valarray<double>;
    using ArrayType = Arr;

  protected:
    size_t _n;
    Arr _xc;
    EllCore _mgr;

    /// @brief Deleted copy assignment operator (non-copyable).
    auto operator=(const EllBase& E) -> EllBase& = delete;

  public:
    /**
     * @brief Construct a new EllBase object from a vector and an array.
     *
     * @param[in] val A vector of double values.
     * @param[in] x An array of type Arr. This parameter is moved.
     */
    EllBase(const Vec& val, Arr x)
        : _n{static_cast<std::size_t>(x.size())}, _xc{std::move(x)}, _mgr(val, _n) {}

    /**
     * @brief Construct a new EllBase object from an alpha value and an array.
     *
     * @param[in] alpha A double value representing the scaling factor.
     * @param[in] x An array of type Arr. This parameter is moved.
     */
    EllBase(const double alpha, Arr x)
        : _n{static_cast<std::size_t>(x.size())}, _xc{std::move(x)}, _mgr(alpha, _n) {}

    /**
     * @brief Construct a new EllBase object (move constructor)
     *
     * @param[in] E The parameter "E" is an rvalue reference to an object of type "EllBase".
     */
    EllBase(EllBase&& E) noexcept = default;

    /**
     * @brief Move assignment operator.
     *
     * @param[in] E The parameter "E" is an rvalue reference to an object of type "EllBase".
     * @return EllBase& Reference to this object.
     */
    auto operator=(EllBase&& E) noexcept -> EllBase& = default;

    /**
     * @brief Destroy the EllBase object
     */
    ~EllBase() = default;

    /**
     * @brief Construct a new EllBase object (explicit copy)
     *
     * @param[in] E The parameter "E" is a reference to an object of type "EllBase".
     */
    explicit EllBase(const EllBase& E) = default;

    /**
     * @brief Get the center of the ellipsoid.
     *
     * @return Arr The center of the ellipsoid.
     */
    auto xc() const -> Arr { return this->_xc; }

    /**
     * @brief Set the center of the ellipsoid.
     *
     * @param[in] xc The new center of the ellipsoid.
     */
    void set_xc(const Arr& xc) { this->_xc = xc; }

    /**
     * @brief Get the squared radius of the ellipsoid.
     *
     * @return double The squared radius.
     */
    constexpr auto tsq() const -> double { return this->_mgr.tsq(); }

    /**
     * @brief Set whether to use parallel cut.
     *
     * @param[in] value True to use parallel cut, false otherwise.
     */
    void set_use_parallel_cut(bool value) { this->_mgr.set_use_parallel_cut(value); }

    /**
     * @brief Update ellipsoid using a deep cut.
     *
     * @f[
     *     g^T (x - x_c) + \beta \le 0
     * @f]
     *
     * @dot
     *   digraph bias_cut_update {
     *     bgcolor="transparent";
     *     node [shape=box, style=filled, fillcolor="#d4e6f1"];
     *     cut [label="Cut: g, beta", fillcolor="#a9cce3"];
     *     update [label="x_c -= g"];
     *     check [label="Success?", shape=diamond, fillcolor="#f9e79f"];
     *     done [label="Updated\nellipsoid", fillcolor="#7fb3d8"];
     *     fail [label="NoEffect", fillcolor="#fadbd8"];
     *     cut -> update;
     *     update -> check;
     *     check -> done [label="Yes", color="#27ae60"];
     *     check -> fail [label="No", color="#e74c3c"];
     *   }
     * @enddot
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_bias_cut(const std::pair<Arr, T>& cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec& grad, const T& beta) {
            if constexpr (Stable) {
                return this->_mgr.update_stable_bias_cut(grad, beta);
            } else {
                return this->_mgr.update_bias_cut(grad, beta);
            }
        });
    }

    /**
     * @brief Update ellipsoid using a central cut.
     *
     * @f[
     *     g^T (x - x_c) \le 0
     * @f]
     *
     * @dot
     *   digraph central_cut_update {
     *     bgcolor="transparent";
     *     node [shape=box, style=filled, fillcolor="#d4e6f1"];
     *     cut [label="Cut: g, beta=0", fillcolor="#a9cce3"];
     *     update [label="x_c -= g"];
     *     check [label="Success?", shape=diamond, fillcolor="#f9e79f"];
     *     done [label="Updated\nellipsoid", fillcolor="#7fb3d8"];
     *     fail [label="NoEffect", fillcolor="#fadbd8"];
     *     cut -> update;
     *     update -> check;
     *     check -> done [label="Yes", color="#27ae60"];
     *     check -> fail [label="No", color="#e74c3c"];
     *   }
     * @enddot
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_central_cut(const std::pair<Arr, T>& cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec& grad, const T& beta) {
            if constexpr (Stable) {
                return this->_mgr.update_stable_central_cut(grad, beta);
            } else {
                return this->_mgr.update_central_cut(grad, beta);
            }
        });
    }

    /**
     * @brief Update ellipsoid using a cut with a specific Q matrix.
     *
     * @f[
     *     Q^+ = Q - \frac{\sigma}{\omega} Q g g^T Q, \qquad
     *     \kappa^+ = \kappa \cdot \delta
     * @f]
     *
     * @dot
     *   digraph q_cut_update {
     *     bgcolor="transparent";
     *     node [shape=box, style=filled, fillcolor="#d4e6f1"];
     *     cut [label="Cut: g, beta", fillcolor="#a9cce3"];
     *     q_update [label="Q += sigma/omega\n* Q*g*g^T*Q"];
     *     kappa [label="kappa *= delta"];
     *     xc [label="x_c -= g"];
     *     check [label="Success?", shape=diamond, fillcolor="#f9e79f"];
     *     done [label="Updated\nellipsoid", fillcolor="#7fb3d8"];
     *     fail [label="NoEffect", fillcolor="#fadbd8"];
     *     cut -> q_update -> kappa -> xc -> check;
     *     check -> done [label="Yes", color="#27ae60"];
     *     check -> fail [label="No", color="#e74c3c"];
     *   }
     * @enddot
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_q(const std::pair<Arr, T>& cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec& grad, const T& beta) {
            if constexpr (Stable) {
                return this->_mgr.update_stable_q(grad, beta);
            } else {
                return this->_mgr.update_q(grad, beta);
            }
        });
    }

  protected:
    /**
     * @brief Update ellipsoid core function using the cut(s).
     *
     * @tparam T Type of the beta parameter.
     * @tparam Fn Type of the cut strategy function.
     * @param[in] cut A pair containing the gradient and beta value.
     * @param[in] cut_strategy The strategy function to apply the cut.
     * @return CutStatus The status of the cut.
     */
    template <typename T, typename Fn>
    auto _update_core(const std::pair<Arr, T>& cut, Fn&& cut_strategy) -> CutStatus {
        const auto& grad = cut.first;
        const auto& beta = cut.second;
        std::valarray<double> g(this->_n);
        for (size_t i = 0; i != this->_n; ++i) {
            g[i] = grad[i];
        }

        auto result = cut_strategy(g, beta);

        if (result == CutStatus::Success) {
            for (size_t i = 0; i != this->_n; ++i) {
                this->_xc[i] -= g[i];
            }
        }

        return result;
    }
};  // } EllBase
