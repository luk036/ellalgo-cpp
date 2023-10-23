// -*- coding: utf-8 -*-
#pragma once

// #include <limits>
#include <cstddef>
#include <tuple>
#include <valarray>

// Modified from CVX code by Almir Mutapcic in 2006.
// Adapted in 2010 for impulse response peak-minimization by convex iteration by
// Christine Law.
//
// "FIR Filter Design via Spectral Factorization and Convex Optimization"
// by S.-P. Wu, S. Boyd, and L. Vandenberghe
//
// Designs an FIR lowpass filter using spectral factorization method with
// constraint on maximum passband ripple and stopband attenuation:
//
//   minimize   max |H(w)|                      for w in stopband
//       s.t.   1/delta <= |H(w)| <= delta      for w in passband
//
// We change variables via spectral factorization method and get:
//
//   minimize   max R(w)                          for w in stopband
//       s.t.   (1/delta)**2 <= R(w) <= delta**2  for w in passband
//              R(w) >= 0                         for all w
//
// where R(w) is squared magnitude frequency response
// (and Fourier transform of autocorrelation coefficients r).
// Variables are coeffients r and G = hh' where h is impulse response.
// delta is allowed passband ripple.
// This is a convex problem (can be formulated as an SDP after sampling).

// *********************************************************************
// filter specs (for a low-pass filter)
// *********************************************************************
// number of FIR coefficients (including zeroth)

/*!
 * @brief Oracle for FIR lowpass filter design.
 *
 *    This example is taken from Almir Mutapcic in 2006:
 *
 *        min   \gamma
 *        s.t.  L^2(\omega) \le R(\omega) \le U^2(\omega), \forall \omega \in
 * [0, \pi] R(\omega) > 0, \forall \omega \in [0, \pi]
 */
class LowpassOracle {
    using Vec = std::valarray<double>;
    using Mat = std::valarray<Vec>;
    using ParallelCut = std::pair<Vec, Vec>;

    Mat A;
    double Lpsq;
    double Upsq;
    size_t nwpass;
    size_t nwstop;
    size_t idx1;
    size_t idx2;
    size_t idx3;
    double _fmax = -1e100;
    size_t _kmax = 0;

  public:
    bool more_alt{true};

    /*!
     * @brief Construct a new lowpass oracle object
     *
     * The constructor of the `LowpassOracle` class. It initializes an instance of the
     * `LowpassOracle` class with the given parameters.
     */
    LowpassOracle(size_t N, double wpass, double wstop, double Lpsq, double Upsq);

    /*!
     * @brief
     *
     */
    auto assess_feas(const Vec &x, double &Spsq) -> ParallelCut *;

    /*!
     * @brief
     *
     */
    auto assess_optim(const Vec &x, double &Spsq) -> std::tuple<ParallelCut, bool>;

    /*!
     * @brief
     *
     * @param[in] x
     * @param[in] Spsq
     * @return auto
     */
    auto operator()(const Vec &x, double &Spsq) -> std::tuple<ParallelCut, bool> {
        return this->assess_optim(x, Spsq);
    }
};

// Filter specs
inline auto create_lowpass_case(size_t N = 48) -> std::pair<LowpassOracle, double> {
    const auto delta0_wpass = 0.125;
    const auto delta0_wstop = 0.125;
    // maximum passband ripple in dB (+/- around 0 dB)
    const auto delta1 = 20.0 * std::log10(1.0 + delta0_wpass);
    // stopband attenuation desired in dB
    const auto delta2 = 20.0 * std::log10(delta0_wstop);

    // *********************************************************************
    // optimization parameters
    // *********************************************************************
    // rule-of-thumb discretization (from Cheney's Approximation Theory)
    const auto Lp = std::pow(10, -delta1 / 20);
    const auto Up = std::pow(10, +delta1 / 20);
    const auto Sp = std::pow(10, +delta2 / 20);

    auto omega = LowpassOracle(N, Lp * Lp, Up * Up, 0.12, 0.20);
    return {std::move(omega), Sp * Sp};
}
