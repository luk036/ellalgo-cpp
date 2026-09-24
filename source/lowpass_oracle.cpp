/**
 * @file lowpass_oracle.cpp
 * @brief Implementation of lowpass filter design oracle
 *
 * This file implements an oracle for designing lowpass digital filters
 * using the ellipsoid algorithm. The oracle assesses feasibility and
 * optimality conditions for filter specifications.
 */

#include <cmath>                               // for pow, log10, M_PI, cos
#include <cstddef>                             // for size_t
#include <ellalgo/oracles/lowpass_oracle.hpp>  // for LowpassOracle, filter_...
#include <numbers>
#include <tuple>     // for tuple
#include <valarray>  // for valarray

using Vec = std::valarray<double>;
using Mat = std::valarray<Vec>;
using ParallelCut = std::pair<Vec, Vec>;

#ifndef M_PI
constexpr double M_PI = std::numbers::pi;
#endif

/**
 * The above function is a constructor for a lowpass filter design class that initializes various
 * parameters and matrices used in the filter design process.
 *
 * @param[in] N The parameter N represents the order of the filter. It determines the number of
 * coefficients used in the filter design.
 * @param[in] Lpsq Lpsq is the squared lower passband edge frequency. It represents the frequency
 * below which the filter allows all signals to pass through without attenuation.
 * @param[in] Upsq Upsq is the upper squared frequency limit for the lowpass filter. It represents
 * the maximum frequency that the filter allows to pass through without significant attenuation.
 * @param[in] wpass The parameter "wpass" represents the normalized passband frequency. It is used
 * in the filter design process to determine the number of frequency points within the passband.
 * @param[in] wstop The parameter "wstop" represents the stopband edge frequency in the filter
 * design process. It is a value between 0 and 1, where 1 corresponds to the Nyquist frequency.
 */
LowpassOracle::LowpassOracle(size_t N, double Lpsq, double Upsq, double wpass, double wstop)
    : Lpsq{Lpsq}, Upsq{Upsq} {
    // *********************************************************************
    // optimization parameters
    // *********************************************************************
    // rule-of-thumb discretization (from Cheney's Approximation Theory)
    const auto m = 15 * N;
    Vec w(m);
    for (size_t i = 0U; i != m; ++i) {
        w[i] = static_cast<double>(i) * M_PI / static_cast<double>(m - 1);
    }

    // A is the matrix used to compute the power spectrum
    // A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos((N-1)*w)]
    this->A = Mat(Vec(N + 1), m);
    for (auto i = 0U; i != m; ++i) {
        this->A[i][0] = 1.0;
        for (auto j = 1U; j != N + 1; ++j) {
            this->A[i][j] = 2.0 * std::cos(w[i] * j);
        }
    }
    this->nwpass = static_cast<int>(std::floor(wpass * static_cast<double>(m - 1)) + 1);
    this->nwstop = static_cast<int>(std::floor(wstop * static_cast<double>(m - 1)) + 1);

    // For round robin
    this->_rr1 = RoundRobin(0, static_cast<std::size_t>(this->nwpass));
    this->_rr2 = RoundRobin(static_cast<std::size_t>(this->nwpass),
                            static_cast<std::size_t>(this->nwstop));
    this->_rr3 = RoundRobin(static_cast<std::size_t>(this->nwstop), this->A.size());
}

/**
 * The function assess_feas in the LowpassOracle class assesses the optimization of a given input
 * vector x based on various constraints and returns a tuple containing the gradient and objective
 * function values, along with a boolean indicating whether the optimization is complete.
 *
 * @param[in] x A 1-dimensional array representing the optimization variables.
 * @param[in, out] Spsq Spsq is a reference to a double variable. It is used to store the maximum
 * value of the stopband constraint.
 *
 * @return The function `assess_feas` returns a tuple containing a `ParallelCut` object and a
 * boolean value.
 */
auto LowpassOracle::assess_feas(const Vec& x, const double& Spsq) -> ParallelCut* {
    auto& cut = this->_cut;

    if (this->scan_band(x, this->_rr1, 0U, static_cast<size_t>(this->nwpass), this->Lpsq, true,
                        this->Upsq, false, cut)) {
        return &cut;
    }
    if (this->scan_band(x, this->_rr3, static_cast<size_t>(this->nwstop), this->A.size(), 0.0, true,
                        Spsq, true, cut)) {
        return &cut;
    }
    if (this->scan_band(x, this->_rr2, static_cast<size_t>(this->nwpass),
                        static_cast<size_t>(this->nwstop), 0.0, false, 0.0, false, cut)) {
        return &cut;
    }

    if (x[0] < 0.0) {
        Vec g(0.0, x.size());
        g[0] = -1.0;
        cut.second = Vec{-x[0]};
        cut.first = g;
        return &cut;
    }

    return nullptr;
}

auto LowpassOracle::scan_band(const Vec& x, RoundRobin& rr, size_t lo, size_t hi, double lower,
                              bool has_upper, double upper, bool track_max,
                              ParallelCut& cut) -> bool {
    if (track_max) {
        this->_fmax = -1e100;  // std::numeric_limits<double>::min()
        this->_kmax = -1;
    }

    for (size_t k = 0; k != hi - lo; ++k) {
        const auto idx = rr.next();
        auto v = 0.0;
        for (size_t j = 0U; j != x.size(); ++j) {
            v += this->A[idx][j] * x[j];
        }

        if (has_upper && v > upper) {
            cut.second = Vec{v - upper, v - lower};
            cut.first = this->A[idx];
            return true;
        }
        if (v < lower) {
            cut.second = has_upper ? Vec{lower - v, upper - v} : Vec{lower - v};
            cut.first = -this->A[idx];
            return true;
        }
        if (track_max && v > this->_fmax) {
            this->_fmax = v;
            this->_kmax = static_cast<int>(idx);
        }
    }
    return false;
}

/**
 * The function assess_optim in the LowpassOracle class assesses the optimization of a given input
 * vector x based on various constraints and returns a tuple containing the gradient and objective
 * function values, along with a boolean indicating whether the optimization is complete.
 *
 * @param[in] x A 1-dimensional array representing the optimization variables.
 * @param[in] Spsq Spsq is a reference to a double variable. It is used to store the maximum value
 * of the stopband constraint.
 *
 * @return The function `assess_optim` returns a tuple containing a `ParallelCut` object and a
 * boolean value.
 */
auto LowpassOracle::assess_optim(const Vec& x, double& Spsq) -> std::tuple<ParallelCut, bool> {
    auto* cut = this->assess_feas(x, Spsq);
    if (cut != nullptr) {
        return {*cut, false};
    }
    // Begin objective function
    Spsq = this->_fmax;  // output
    return {{this->A[static_cast<size_t>(this->_kmax)], Vec{0.0, this->_fmax}}, true};
}
