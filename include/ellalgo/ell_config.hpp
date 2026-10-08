/**
 * @file ell_config.hpp
 * @brief Configuration types and constants for the ellipsoid algorithm
 */

#pragma once

#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>  // for pair

/**
 * @brief Configuration options for the ellipsoid algorithm
 *
 * This structure contains the configuration parameters for controlling
 * the behavior of the ellipsoid algorithm, including maximum iterations
 * and convergence tolerance.
 */
struct Options {
    size_t max_iters;  ///< Maximum number of iterations allowed
    double tolerance;  ///< Convergence tolerance for stopping criteria
    bool verbose;      ///< Enable iteration logging

    /**
     * @brief Default constructor
     *
     * Initializes with default values: max_iters = 2000, tolerance = 1e-20,
     * verbose = false.
     */
    Options() : max_iters{2000}, tolerance{1e-20}, verbose{false} {}

    /**
     * @brief Constructor with custom parameters
     *
     * @param[in] max_iters Maximum number of iterations
     * @param[in] tol Convergence tolerance
     */
    Options(size_t max_iters, double tol) : max_iters{max_iters}, tolerance{tol}, verbose{false} {}
};

/**
 * @brief Status of cutting plane operations
 *
 * This enumeration represents the possible outcomes of cutting plane
 * operations in the ellipsoid algorithm.
 */
enum class CutStatus {
    Success,   ///< Cut was successful and ellipsoid was updated
    NoSoln,    ///< No solution exists (infeasible)
    NoEffect,  ///< Cut had no effect on ellipsoid
    Unknown    ///< Unknown status
};

/// @brief Stream output operator for CutStatus
inline auto operator<<(std::ostream& os, CutStatus s) -> std::ostream& {
    switch (s) {
        case CutStatus::Success:
            return os << "✓ success";
        case CutStatus::NoSoln:
            return os << "✗ no solution";
        case CutStatus::NoEffect:
            return os << "⏭ no effect";
        case CutStatus::Unknown:
            return os << "? unknown";
    }
    return os;
}

/**
 * @brief Result of a cutting-plane calculation
 *
 * POD struct replacing nested `std::tuple<CutStatus, tuple<double,double,double>>`.
 * Enables register-passing and better inlining on all modern ABIs.
 */
struct CutResult {
    CutStatus status;  ///< Status of the cut
    double rho;        ///< Step size along gradient direction
    double sigma;      ///< Scaling factor for matrix update
    double delta;      ///< Contraction factor for ellipsoid volume
};

/**
 * @brief Raised by an iterative solver that fails to converge in its budget
 *
 * Derives from `std::runtime_error` so existing `catch (const std::runtime_error&)`
 * handlers keep working while callers can catch this precise type.
 */
class ConvergenceError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Termination status of a cutting-plane / binary-search solve
 */
enum class SolverStatus {
    Success,     ///< A solution (or converged bracket) was produced
    Infeasible,  ///< Search space exhausted; no solution exists
    MaxIters     ///< Iteration cap reached before terminating
};

/**
 * @brief Result of a cutting-plane / binary-search solve
 *
 * Derives from `std::tuple<X, std::size_t>` so existing `std::get<0>` /
 * `std::get<1>` access and 2-element structured bindings keep working, while
 * additionally exposing `.status` to tell an exhausted search space
 * (`Infeasible`) apart from an iteration-cap stop (`MaxIters`).
 *
 * @tparam X Solution / best-so-far value type
 */
template <typename X> struct SolverResult : std::tuple<X, std::size_t> {
    SolverStatus status;

    SolverResult(X x, std::size_t niter, SolverStatus s)
        : std::tuple<X, std::size_t>(std::move(x), niter), status(s) {}
};

namespace std {
    template <typename X> struct tuple_size<SolverResult<X>> : integral_constant<size_t, 2> {};

    template <typename X> struct tuple_element<0, SolverResult<X>> {
        using type = X;
    };

    template <typename X> struct tuple_element<1, SolverResult<X>> {
        using type = std::size_t;
    };
}  // namespace std

/**
 * @brief Single cut parameter β in gᵀ(x - xc) + β ≤ 0
 *
 * Represents the bias term in a cutting plane constraint.
 *
 * In Rust this is a newtype: `pub struct SingleCut(pub f64);`
 * In Python this is a type alias: `SingleCut = float`
 * In C++ this is a type alias for consistency:
 */
using SingleCut = double;

// --- C++20 Concepts (simple constraints to avoid MSVC ICE) ---
#if __cpp_concepts >= 201907L
#    include <concepts>

/**
 * @note These concepts formalize the Strategy pattern contracts at compile
 *       time: any type satisfying OracleFeas (assess_feas) can be injected
 *       into the cutting-plane Context as a Strategy. SearchSpace is the
 *       Context-side contract (xc, tsq, update_bias_cut, update_central_cut).
 */
template <typename O, typename A>
concept OracleFeas = requires(O& o, const A& x) {
    { o.assess_feas(x) };
};

template <typename O, typename A, typename N>
concept OracleOptim = requires(O& o, const A& x, N& g) {
    { o.assess_optim(x, g) };
};

template <typename O, typename A, typename N>
concept OracleOptimQ = requires(O& o, const A& x, N& g, bool r) {
    { o.assess_optim_q(x, g, r) };
};

template <typename O, typename N>
concept OracleBS = requires(O& o, N& g) {
    { o.assess_bs(g) };
};

template <typename S>
concept SearchSpace = requires(S& s, const std::pair<typename S::ArrayType, double>& cut) {
    typename S::ArrayType;
    { s.xc() };
    { s.tsq() };
    { s.update_bias_cut(cut) } -> std::same_as<CutStatus>;
    { s.update_central_cut(cut) } -> std::same_as<CutStatus>;
};

#endif
