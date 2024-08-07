// -*- coding: utf-8 -*-
#include <ellalgo/cutting_plane.hpp>          // for cutting_plane_optim
#include <ellalgo/ell.hpp>                    // for Ell
#include <ellalgo/oracles/lowpass_oracle.hpp> // for LowpassOracle
#include <ellalgo/utility.hpp>                // for zeros
// #include <xtensor-blas/xlinalg.hpp>
#include <cmath>                       // for pow, log10, acos, cos
#include <tuple>                       // for get, make_tuple
#include <vector>                      // for vector, vector<>::size...
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xbroadcast.hpp>      // for xbroadcast
#include <xtensor/xbuilder.hpp>        // for concatenate, linspace
#include <xtensor/xcontainer.hpp>      // for xcontainer
#include <xtensor/xexception.hpp>      // for throw_concatenate_error
#include <xtensor/xgenerator.hpp>      // for xgenerator
#include <xtensor/xlayout.hpp>         // for layout_type, layout_ty...
#include <xtensor/xoperation.hpp>      // for xfunction_type_t, oper...
#include <xtensor/xslice.hpp>          // for xtuph, all, range, _
#include <xtensor/xtensor_forward.hpp> // for xarray
#include <xtensor/xutils.hpp>          // for accumulate
#include <xtensor/xview.hpp>           // for view, xview

#include "benchmark/benchmark.h"  // for BENCHMARK, State, BENC...
#include "ellalgo/ell_config.hpp" // for CInfo, Options

using Arr = xt::xarray<double, xt::layout_type::row_major>;

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
// Variables are coeffients r and gra = hh' where h is impulse response.
// delta is allowed passband ripple.
// This is a convex problem (can be formulated as an SDP after sampling).

// rand('twister',sum(100*clock))
// randn('state',sum(100*clock))

// *********************************************************************
// filter specs (for a low-pass filter)
// *********************************************************************
// number of FIR coefficients (including zeroth)
struct filter_design_construct {
  const int N = 32;
  Arr Ap;
  Arr As;
  Arr Anr;
  double Lpsq;
  double Upsq;
  double Spsq;

  filter_design_construct() {
    static const auto PI = std::acos(-1);

    const auto wpass = 0.12 * PI; // end of passband
    const auto wstop = 0.20 * PI; // start of stopband
    const auto delta0_wpass = 0.125;
    const auto delta0_wstop = 0.125;
    // maximum passband ripple in dB (+/- around 0 dB)
    const auto delta = 20 * std::log10(1 + delta0_wpass);
    // stopband attenuation desired in dB
    const auto delta2 = 20 * std::log10(delta0_wstop);

    // *********************************************************************
    // optimization parameters
    // *********************************************************************
    // rule-of-thumb discretization (from Cheney's Approximation Theory)
    const auto m = 15 * N;
    const Arr w{xt::linspace<double>(0, PI, m)}; // omega

    // passband 0 <= w <= w_pass
    const auto Lp = std::pow(10, -delta / 20);
    const auto Up = std::pow(10, +delta / 20);

    // A is the matrix used to compute the power spectrum
    // A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos((N-1)*w)]

    // Arr An = 2 * xt::cos(xt::linalg::outer(w, xt::arange(1, N)));
    Arr An = xt::zeros<double>({m, N - 1});
    for (auto i = 0; i != m; ++i) {
      for (auto j = 0; j != N - 1; ++j) {
        An(i, j) = 2.0 * std::cos(w(i) * (j + 1));
      }
    }
    Arr A = xt::concatenate(xt::xtuple(xt::ones<double>({m, 1}), An), 1);

    const auto ind_p = xt::where(w <= wpass)[0]; // passband
    Ap = xt::view(A, xt::range(0, ind_p.size()), xt::all());

    // stopband (w_stop <= w)
    auto ind_s = xt::where(wstop <= w)[0]; // stopband
    const auto Sp = std::pow(10, delta2 / 20);

    using xt::placeholders::_;
    As = xt::view(A, xt::range(ind_s[0], _), xt::all());

    // remove redundant contraints
    // ind_nr = setdiff(1:m,ind_p)   // fullband less passband
    // ind_nr = setdiff(ind_nr, ind_s) // luk: for making parallel cut
    // auto ind_nr = np.setdiff1d(xt::arange(m), ind_p);
    // auto ind_nr = np.setdiff1d(ind_nr, ind_s);
    auto ind_beg = ind_p[ind_p.size() - 1];
    auto ind_end = ind_s[0];
    Anr = xt::view(A, xt::range(ind_beg + 1, ind_end), xt::all());

    Lpsq = Lp * Lp;
    Upsq = Up * Up;
    Spsq = Sp * Sp;
  }
};

// ********************************************************************
// optimization
// ********************************************************************

auto run_lowpass(bool use_parallel_cut) {
  static const filter_design_construct Fdc{};

  auto r0 = zeros({Fdc.N}); // initial x0
  Ell<Arr> ellip(40.0, r0);
  LowpassOracle omega(Fdc.Ap, Fdc.As, Fdc.Anr, Fdc.Lpsq, Fdc.Upsq);
  Options options{};

  options.max_iters = 50000;
  ellip.set_use_parallel_cut(use_parallel_cut);
  // options.tolerance = 1e-8;

  auto t = Fdc.Spsq;
  const auto __result = cutting_plane_optim(omega, ellip, t, options);
  const auto &r = std::get<0>(__result);
  const auto &num_iters = std::get<1>(__result);
  // std::cout << "lowpass r: " << r << '\n';
  // auto Ustop = 20 * std::log10(std::sqrt(Spsq_new));
  // std::cout << "Min attenuation in the stopband is " << Ustop << " dB.\n";
  // CHECK(r[0] >= 0.0);
  return std::make_tuple(ell_info.feasible, r.size() != 0U);
}

/**
 * @brief
 *
 * @param[in,out] state
 */
static void Lowpass_with_parallel_cut(benchmark::State &state) {
  while (state.KeepRunning()) {
    run_lowpass(true);
  }
}

// Register the function as a benchmark
BENCHMARK(Lowpass_with_parallel_cut);

/**
 * @brief
 *
 * @param[in,out] state
 */
static void Lowpass_without_parallel_cut(benchmark::State &state) {
  while (state.KeepRunning()) {
    run_lowpass(false);
  }
}

// Register the function as a benchmark
BENCHMARK(Lowpass_without_parallel_cut);

BENCHMARK_MAIN();
