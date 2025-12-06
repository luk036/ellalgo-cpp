#pragma once

#include <cstddef>
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

    /**
     * @brief Default constructor
     *
     * Initializes with default values: max_iters = 2000, tolerance = 1e-20
     */
    Options() : max_iters{2000}, tolerance{1e-20} {}

    /**
     * @brief Constructor with custom parameters
     *
     * @param[in] max_iters Maximum number of iterations
     * @param[in] tol Convergence tolerance
     */
    Options(size_t max_iters, double tol) : max_iters{max_iters}, tolerance{tol} {}
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

/**
 * @brief Information about the computation result
 *
 * This structure contains information about the computation,
 * including feasibility status and number of iterations used.
 */
struct CInfo {
    bool feasible;      ///< Whether a feasible solution was found
    size_t num_iters;   ///< Number of iterations performed
};

/**
 * @brief Type alias for the array type used by template parameter T
 *
 * This type alias extracts the ArrayType nested type from template parameter T.
 *
 * @tparam T The type containing ArrayType
 */
template <typename T> using ArrayType = typename T::ArrayType;

/**
 * @brief Type alias for the cut choice type used by template parameter T
 *
 * This type alias extracts the CutChoice nested type from template parameter T.
 *
 * @tparam T The type containing CutChoice
 */
template <typename T> using CutChoice = typename T::CutChoice;

/**
 * @brief Type alias for a cutting plane concept
 *
 * This type alias defines a pair of array and cut choice, representing
 * a cutting plane in the algorithm.
 *
 * @tparam T The template parameter type
 */
template <typename T> using CutConcept = std::pair<ArrayType<T>, CutChoice<T>>;

/**
 * @brief Type alias for return type of Q optimization
 *
 * This type alias defines the return type for Q optimization functions,
 * containing cut concept, boolean flags, and array data.
 *
 * @tparam T The template parameter type
 */
template <typename T> using RetQ = std::tuple<CutConcept<T>, bool, ArrayType<T>, bool>;

#if __cpp_concepts >= 201907L
// #include "ell_concepts.hpp"
#endif
