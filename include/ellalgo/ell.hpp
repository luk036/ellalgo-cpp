#pragma once

#include <valarray>

#include "ell_config.hpp"
#include "ell_core.hpp"

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 * The `Ell` class represents an ellipsoid search space:
 *
 *   ell = {x | (x - xc)' mq^-1 (x - xc) ≤ κ}
 *
 * It is used to define and manipulate ellipsoids in a multidimensional space.
 * The ellipsoid is defined by a center point (`_xc`) and a core function
 * (`_mgr`). The core function is responsible for updating the ellipsoid based
 * on cutting planes. The `Ell` class provides methods to update the ellipsoid
 * using different types of cutting planes and to retrieve information about the
 * ellipsoid, such as the center point and the squared radius.
 *
 * This version keeps $Q$ symmetric but no promise of positive definite
 */
template <typename Arr> class Ell {
  public:
    using Vec = std::valarray<double>;
    using ArrayType = Arr;

  private:
    const size_t _n;
    Arr _xc;
    EllCore _mgr;

    /**
     * @brief Deleted copy assignment operator.
     *
     * @param[in] E The parameter "E" is a reference to an object of type "Ell".
     */
    auto operator=(const Ell &E) -> Ell & = delete;

  public:
    /**
     * @brief Construct a new Ell object from a vector and an array.
     *
     * @param[in] val A vector of double values.
     * @param[in] x An array of type Arr.
     */
    Ell(const Vec &val, Arr x) : _n{x.size()}, _xc{std::move(x)}, _mgr(val, _n) {}

    /**
     * @brief Construct a new Ell object from an alpha value and an array.
     *
     * @param[in] alpha A double value representing the scaling factor.
     * @param[in] x An array of type Arr.
     */
    Ell(const double &alpha, Arr x) : _n{x.size()}, _xc{std::move(x)}, _mgr(alpha, _n) {}

    /**
     * @brief Construct a new Ell object (move constructor)
     *
     * @param[in] E The parameter "E" is an rvalue reference to an object of type "Ell".
     */
    Ell(Ell &&E) noexcept = default;

    /**
     * @brief Destroy the Ell object
     *
     */
    ~Ell() = default;

    /**
     * @brief Construct a new Ell object (explicit copy)
     *
     * @param[in] E The parameter "E" is a reference to an object of type "Ell".
     */
    explicit Ell(const Ell &E) = default;

    /**
     * @brief Explicitly copy the Ell object.
     *
     * @return Ell A new Ell object that is a copy of the current object.
     */
    auto copy() const -> Ell { return Ell(*this); }

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
    void set_xc(const Arr &xc) { this->_xc = xc; }

    /**
     * @brief Get the squared radius of the ellipsoid.
     *
     * @return double The squared radius.
     */
    auto tsq() const -> double { return this->_mgr.tsq(); }

    /**
     * @brief Set whether to use parallel cut.
     *
     * @param[in] value True to use parallel cut, false otherwise.
     */
    void set_use_parallel_cut(bool value) { this->_mgr.set_use_parallel_cut(value); }

    /**
     * @brief Update ellipsoid using a deep cut.
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_bias_cut(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec &grad, const T &beta) {
            return this->_mgr.update_bias_cut(grad, beta);
        });
    }

    /**
     * @brief Update ellipsoid using a central cut.
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_central_cut(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec &grad, const T &beta) {
            return this->_mgr.update_central_cut(grad, beta);
        });
    }

    /**
     * @brief Update ellipsoid using a cut with a specific Q matrix.
     *
     * @tparam T Type of the beta parameter.
     * @param[in] cut A pair containing the gradient and beta value.
     * @return CutStatus The status of the cut.
     */
    template <typename T> auto update_q(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(
            cut, [this](Vec &grad, const T &beta) { return this->_mgr.update_q(grad, beta); });
    }

  private:
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
    auto _update_core(const std::pair<Arr, T> &cut, Fn &&cut_strategy) -> CutStatus {
        const auto &grad = cut.first;
        const auto &beta = cut.second;
        std::valarray<double> g(this->_n);
        for (auto i = 0U; i != this->_n; ++i) {
            g[i] = grad[i];
        }

        auto result = cut_strategy(g, beta);

        if (result == CutStatus::Success) {
            for (auto i = 0U; i != this->_n; ++i) {
                this->_xc[i] -= g[i];
            }
        }

        return result;
    }
};  // } Ell