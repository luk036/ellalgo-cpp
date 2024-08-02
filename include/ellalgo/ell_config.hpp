#pragma once

#include <cstddef>
#include <utility>  // for pair

/**
 * @brief Options
 *
 */
struct Options {
    size_t max_iters;
    double tolerance;

    Options() : max_iters{2000}, tolerance{1e-20} {}
    Options(size_t max_iters, double tol) : max_iters{max_iters}, tolerance{tol} {}
};

/**
 * @brief Cut Status
 *
 */
enum class CutStatus { Success, NoSoln, NoEffect, Unknown };

/**
 * @brief CInfo
 *
 */
struct CInfo {
    bool feasible;
    size_t num_iters;
};

template <typename T> using ArrayType = typename T::ArrayType;
template <typename T> using CutChoice = typename T::CutChoice;
template <typename T> using CutConcept = std::pair<ArrayType<T>, CutChoice<T>>;
template <typename T> using RetQ = std::tuple<CutConcept<T>, bool, ArrayType<T>, bool>;

#if __cpp_concepts >= 201907L
// #include "ell_concepts.hpp"
#endif
