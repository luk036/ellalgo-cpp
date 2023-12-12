#include <stddef.h>  // for size_t

#include <cmath>                               // for pow, log10, M_PI, cos
#include <ellalgo/oracles/lowpass_oracle.hpp>  // for LowpassOracle, filter_...
#include <tuple>                               // for tuple
#include <type_traits>                         // for move
#include <valarray>                            // for valarray
#include <vector>                              // for vector, vector<>::size_...

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
 * @param[in] N The parameter N represents the order of the filter. It determines the number of
 * coefficients used in the filter design.
 * @param[in] Lpsq Lpsq is the squared lower passband edge frequency. It represents the frequency below
 * which the filter allows all signals to pass through without attenuation.
 * @param[in] Upsq Upsq is the upper squared frequency limit for the lowpass filter. It represents the
 * maximum frequency that the filter allows to pass through without significant attenuation.
 * @param[in] wpass The parameter "wpass" represents the normalized passband frequency. It is used in
 * the filter design process to determine the number of frequency points within the passband.
 * @param[in] wstop The parameter "wstop" represents the stopband edge frequency in the filter design
 * process. It is a value between 0 and 1, where 1 corresponds to the Nyquist frequency.
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

    // For round robin
    this->idx1 = 0U;
    this->idx2 = this->nwpass;
    this->idx3 = this->nwstop;
}

/**
 * The function assess_feas in the LowpassOracle class assesses the optimization of a given input
 * vector x based on various constraints and returns a tuple containing the gradient and objective
 * function values, along with a boolean indicating whether the optimization is complete.
 *
 * @param[in] x A 1-dimensional array representing the optimization variables.
 * @param[in, out] Spsq Spsq is a reference to a double variable. It is used to store the maximum value of
 * the stopband constraint.
 *
 * @return The function `assess_feas` returns a tuple containing a `ParallelCut` object and a
 * boolean value.
 */
auto LowpassOracle::assess_feas(const Vec &x, const double &Spsq) -> ParallelCut * {
    static ParallelCut cut = std::make_pair(Vec{0.0}, Vec{0.0});

    this->more_alt = true;
    auto n = x.size();

    auto matrix_vector = [this, &x](size_t k) {
        double sum = 0.0;
        for (size_t j = 0U; j != x.size(); ++j) {
            sum += this->A[k][j] * x[j];
        }
        return sum;
    };

    // case 2,
    // 2.0 passband constraints
    for (size_t __k = 0; __k != this->nwpass; ++__k) {
        ++this->idx1;
        if (this->idx1 == this->nwpass) {
            this->idx1 = 0;  // round robin
        }
        double v = matrix_vector(this->idx1);
        if (v > this->Upsq) {
            cut.second = Vec{v - this->Upsq, v - this->Lpsq};
            cut.first = this->A[this->idx1];
            return &cut;
        }
        if (v < this->Lpsq) {
            cut.second = Vec{-v + this->Lpsq, -v + this->Upsq};
            cut.first = -this->A[this->idx1];
            return &cut;
        }
    }

    // case 3,
    // 3.0 stopband constraint
    auto N = A.size();
    this->_fmax = -1e100;  // std::numeric_limits<double>::min()
    this->_kmax = 0U;
    for (size_t __k = this->nwstop; __k != N; ++__k) {
        ++this->idx3;
        if (this->idx3 == N) {
            this->idx3 = this->nwstop;  // round robin
        }
        double v = matrix_vector(this->idx3);
        if (v > Spsq) {
            cut.second = Vec{v - Spsq, v};
            cut.first = this->A[this->idx3];
            return &cut;
        }
        if (v < 0.0) {
            cut.second = Vec{-v, -v + Spsq};
            cut.first = -this->A[this->idx3];
            return &cut;
        }
        if (v > this->_fmax) {
            this->_fmax = v;
            this->_kmax = this->idx3;
        }
    }

    // case 4,
    // 1.0 nonnegative-real constraint
    for (size_t __k = this->nwpass; __k != this->nwstop; ++__k) {
        ++this->idx2;
        if (this->idx2 == this->nwstop) {
            this->idx2 = this->nwpass;  // round robin
        }
        double v = matrix_vector(this->idx2);
        if (v < 0.0) {
            cut.second = Vec{-v};
            cut.first = -this->A[this->idx2];
            return &cut;
        }
    }

    this->more_alt = false;

    // 1.0 nonnegative-real constraint
    // case 1,
    if (x[0] < 0.0) {
        Vec g(0.0, n);
        g[0] = -1.0;
        cut.second = Vec{-x[0]};
        cut.first = g;
        return &cut;
    }

    return nullptr;
}

/**
 * The function assess_optim in the LowpassOracle class assesses the optimization of a given input
 * vector x based on various constraints and returns a tuple containing the gradient and objective
 * function values, along with a boolean indicating whether the optimization is complete.
 *
 * @param[in] x A 1-dimensional array representing the optimization variables.
 * @param[in] Spsq Spsq is a reference to a double variable. It is used to store the maximum value of
 * the stopband constraint.
 *
 * @return The function `assess_optim` returns a tuple containing a `ParallelCut` object and a
 * boolean value.
 */
auto LowpassOracle::assess_optim(const Vec &x, double &Spsq) -> std::tuple<ParallelCut, bool> {
    auto cut = this->assess_feas(x, Spsq);
    if (cut) {
        return {*cut, false};
    }
    // Begin objective function
    Spsq = this->_fmax;  // output
    return {{this->A[this->_kmax], Vec{0.0, this->_fmax}}, true};
}
