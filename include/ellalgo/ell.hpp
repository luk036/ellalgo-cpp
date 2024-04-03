// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <tuple>
#include <valarray>

#include "ell_config.hpp"
#include "ell_core.hpp"
#include "ell_matrix.hpp"

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 * The `Ell` class represents an ellipsoid search space:
 *
 *   ell = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
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
     * @brief Construct a new Ell object
     *
     * The `operator=` function is being deleted in this code. This means that the assignment
     * operator is not allowed for objects of the `Ell` class. By deleting this function, the code
     * prevents objects of the `Ell` class from being assigned to each other.
     *
     * @param[in] E The parameter "E" is a reference to an object of type "Ell".
     */
    auto operator=(const Ell &E) -> Ell & = delete;

  public:
    /**
     * @brief Construct a new Ell object
     *
     * The function is a constructor for an Ell object that takes a Vec and an Arr as parameters.
     *
     * @param[in] val The parameter "val" is of type Vec, which is likely a vector or array-like
     * data structure. It is being passed by reference to the constructor of the Ell class.
     * @param[in] x x is an object of type Arr, which is likely an array or vector. It is being
     * passed by value to the constructor of the Ell class.
     */
    Ell(const Vec &val, Arr x) : _n{x.size()}, _xc{std::move(x)}, _mgr(val, _n) {}

    /**
     * @brief Construct a new Ell object
     *
     * The function constructs a new Ell object with a given alpha value and an array of x values.
     *
     * @param[in] alpha The parameter `alpha` is a constant reference to a `double` value. It is
     * used to initialize the `_mgr` member variable of the `Ell` class.
     * @param[in] x The parameter `x` is of type `Arr`, which is likely an array or vector of some
     * kind. It is being passed by value, meaning a copy of the `x` object will be made and stored
     * in the `_xc` member variable of the `Ell` object being constructed.
     */
    Ell(const double &alpha, Arr x) : _n{x.size()}, _xc{std::move(x)}, _mgr(alpha, _n) {}

    /**
     * @brief Construct a new Ell object
     *
     * The function is a constructor for an Ell object that takes an rvalue reference as a
     * parameter.
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
     * @brief Construct a new Ell object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param[in] E The parameter "E" is a reference to an object of type "Ell".
     */
    explicit Ell(const Ell &E) = default;

    /**
     * @brief explicitly copy
     *
     * @return Ell
     */
    auto copy() const -> Ell { return Ell(*this); }

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    auto xc() const -> Arr { return this->_xc; }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    void set_xc(const Arr &xc) { this->_xc = xc; }

    /**
     * @brief
     *
     * @return double
     */
    auto tsq() const -> double { return this->_mgr.tsq(); }

    /**
     * The function sets the value of the use_parallel_cut property in the _mgr object.
     *
     * @param[in] value The value parameter is a boolean value that determines whether or not to use
     * parallel cut.
     */
    void set_use_parallel_cut(bool value) { this->_mgr.set_use_parallel_cut(value); }

    /**
     * @brief Update ellipsoid core function using the deep cut(s)
     *
     * The `update_bias_cut` function is a member function of the `Ell` class. It is used to update
     * the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T> auto update_bias_cut(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec &grad, const T &beta) {
            return this->_mgr.update_bias_cut(grad, beta);
        });
    }

    /**
     * @brief Update ellipsoid core function using the central cut(s)
     *
     * The `update_central_cut` function is a member function of the `Ell` class. It is used to
     * update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T> auto update_central_cut(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(cut, [this](Vec &grad, const T &beta) {
            return this->_mgr.update_central_cut(grad, beta);
        });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `update_q` function is a member function of the `Ell` class. It is used to update the
     * ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T> auto update_q(const std::pair<Arr, T> &cut) -> CutStatus {
        return this->_update_core(
            cut, [this](Vec &grad, const T &beta) { return this->_mgr.update_q(grad, beta); });
    }

  private:
    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `_update_core` function is a private member function of the `Ell` class. It is used to
     * update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
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
