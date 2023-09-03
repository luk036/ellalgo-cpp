#include <stddef.h>    // for size_t
#include <cmath>       // for pow, log10, M_PI, cos
#include <tuple>       // for tuple
#include <type_traits> // for move
#include <vector>      // for vector, vector<>::size_...
#include <valarray>    // for valarray

#include <ellalgo/oracles/lowpass_oracle.hpp>  // for LowpassOracle, filter_...

using Vec = std::valarray<double>;
using Mat = std::valarray<Vec>;
using ParallelCut = std::pair<Vec, Vec>;

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif

/**
 * The above function is a constructor for a lowpass filter design class that initializes various
 * parameters and matrices used in the filter design process.
 * 
 * @param N The parameter N represents the order of the filter. It determines the number of
 * coefficients used in the filter design.
 * @param Lpsq Lpsq is the squared lower passband edge frequency. It represents the frequency below
 * which the filter allows all signals to pass through without attenuation.
 * @param Upsq Upsq is the upper squared frequency limit for the lowpass filter. It represents the
 * maximum frequency that the filter allows to pass through without significant attenuation.
 * @param wpass The parameter "wpass" represents the normalized passband frequency. It is used in the
 * filter design process to determine the number of frequency points within the passband.
 * @param wstop The parameter "wstop" represents the stopband edge frequency in the filter design
 * process. It is a value between 0 and 1, where 1 corresponds to the Nyquist frequency.
 */
LowpassOracle::LowpassOracle(size_t N, double Lpsq, double Upsq, double wpass, double wstop)
    : Lpsq{Lpsq}, Upsq{Upsq}
{
    // *********************************************************************
    // optimization parameters
    // *********************************************************************
    // rule-of-thumb discretization (from Cheney's Approximation Theory)
    const auto m = 15 * N;
    Vec w(m);
    for (size_t i = 0U; i != m; ++i) {
        w[i] = double(i) * M_PI / double(m - 1);
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
    this->nwpass = size_t(std::floor(wpass * double(m - 1)) + 1);
    this->nwstop = size_t(std::floor(wstop * double(m - 1)) + 1);
}

/**
 * The function assess_optim in the LowpassOracle class assesses the optimization of a given input
 * vector x based on various constraints and returns a tuple containing the gradient and objective
 * function values, along with a boolean indicating whether the optimization is complete.
 *
 * @param x A 1-dimensional array representing the optimization variables.
 * @param Spsq Spsq is a reference to a double variable. It is used to store the maximum value of
 * the stopband constraint.
 *
 * @return The function `assess_optim` returns a tuple containing a `ParallelCut` object and a
 * boolean value.
 */
auto LowpassOracle::assess_optim(const Vec &x, double &Spsq) -> std::tuple<ParallelCut, bool> {
    this->more_alt = true;
    auto n = x.size();

    auto matrix_vector = [this, &x](size_t k) {
        double sum = 0.0;
        for (size_t j = 0U; j != x.size(); ++j) {
            sum += this->A[k][j] * x[j];
        }
        return sum;
    };

    // 1.0 nonnegative-real constraint
    // case 1,
    if (x[0] < 0.0) {
        Vec g(0.0, n);
        g[0] = -1.0;
        return {{std::move(g), Vec{-x[0]}}, false};
    }

    // case 2,
    // 2.0 passband constraints
    auto N = A.size();
    for (size_t k = 0; k != this->nwpass; ++k) {
        double v = matrix_vector(k);
        if (v > this->Upsq) {
            Vec f{v - this->Upsq, v - this->Lpsq};
            return {{this->A[k], std::move(f)}, false};
        }
        if (v < this->Lpsq) {
            Vec f{-v + this->Lpsq, -v + this->Upsq};
            return {{-this->A[k], std::move(f)}, false};
        }
    }

    // case 3,
    // 3.0 stopband constraint
    auto fmax = -1.e100;  // std::numeric_limits<double>::min()
    size_t kmax = 0U;
    for (size_t k = this->nwstop; k != N; ++k) {
        double v = matrix_vector(k);
        if (v > Spsq) {
            return {{this->A[k], Vec{v - Spsq, v}}, false};
        }
        if (v < 0.0) {
            return {{-this->A[k], Vec{-v, -v + Spsq}}, false};
        }
        if (v > fmax) {
            fmax = v;
            kmax = k;
        }
    }

    // case 4,
    // 1.0 nonnegative-real constraint
    for (size_t k = this->nwpass; k != nwstop; ++k) {
        double v = matrix_vector(k);
        if (v < 0.0) {
            return {{-this->A[k], Vec{-v}}, false};
        }
    }

    this->more_alt = false;

    // Begin objective function
    Spsq = fmax;  // output
    return {{this->A[kmax], Vec{0.0, fmax}}, true};
}
