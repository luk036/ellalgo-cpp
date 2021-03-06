// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <tuple>

// forward declaration
enum class CUTStatus;

/**
 * @brief Ellipsoid Method for special 1D case
 *
 */
class ell1d {
  public:
    using return_t = std::tuple<CUTStatus, double>;

  private:
    double _r;
    double _xc;

  public:
    /**
     * @brief Construct a new ell1d object
     *
     * @param[in] l
     * @param[in] u
     */
    ell1d(const double& l, const double& u) noexcept : _r{(u - l) / 2}, _xc{l + _r} {}

    /**
     * @brief Construct a new ell1d object
     *
     * @param[in] E
     */
    explicit ell1d(const ell1d& E) = default;

    /**
     * @brief
     *
     * @return double
     */
    auto xc() const noexcept -> double { return _xc; }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    auto set_xc(const double& xc) noexcept -> void { _xc = xc; }

    /**
     * @brief
     *
     * @param[in] cut
     * @return return_t
     */
    auto update(const std::tuple<double, double>& cut) noexcept -> return_t;
};  // } ell1d
