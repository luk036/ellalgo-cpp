#pragma once

#include <cstddef>
#include <ostream>
#include <utility>  // for pair

/**
 * @brief Configuration options for the ellipsoid algorithm
 *
 * This structure contains the configuration parameters for controlling
 * the behavior of the ellipsoid algorithm, including maximum iterations
 * and convergence tolerance.
 */
struct Options {
    size_t max_iters;   ///< Maximum number of iterations allowed
    double tolerance;   ///< Convergence tolerance for stopping criteria
    bool verbose;       ///< Enable iteration logging

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

/// Stream output for CutStatus
inline auto operator<<(std::ostream& os, CutStatus s) -> std::ostream& {
    switch (s) {
        case CutStatus::Success: return os << "✓ success";
        case CutStatus::NoSoln:  return os << "✗ no solution";
        case CutStatus::NoEffect: return os << "⏭ no effect";
        case CutStatus::Unknown: return os << "? unknown";
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
 * @brief Information about the computation result
 *
 * This structure contains information about the computation,
 * including feasibility status and number of iterations used.
 */
struct CInfo {
    bool feasible;     ///< Whether a feasible solution was found
    size_t num_iters;  ///< Number of iterations performed
};

/**
 * @brief Type alias for the array type used by template parameter T
 *
 * @tparam T The type containing ArrayType
 */
template <typename T> using ArrayType = typename T::ArrayType;

/**
 * @brief Type alias for the cut choice type used by template parameter T
 *
 * @tparam T The type containing CutChoice
 */
template <typename T> using CutChoice = typename T::CutChoice;

/**
 * @brief Type alias for a cutting plane concept
 *
 * @tparam T The template parameter type
 */
template <typename T> using CutConcept = std::pair<ArrayType<T>, CutChoice<T>>;

/**
 * @brief Type alias for return type of Q optimization
 *
 * @tparam T The template parameter type
 */
template <typename T> using RetQ = std::tuple<CutConcept<T>, bool, ArrayType<T>, bool>;

/// Single cut parameter β in gᵀ(x - xc) + β ≤ 0
using SingleCut = double;

// --- C++20 Concepts (simple constraints to avoid MSVC ICE) ---
#if __cpp_concepts >= 201907L
#include <concepts>

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
