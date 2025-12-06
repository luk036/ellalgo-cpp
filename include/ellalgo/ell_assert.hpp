/**
 * @file ell_assert.hpp
 * @brief Branch prediction macros for performance optimization
 *
 * This header provides macros for giving hints to the compiler about
 * the likelihood of certain conditions being true or false. This can
 * help the CPU generate more efficient code by optimizing branch
 * prediction.
 */

#pragma once

/**
 * @def ELL_LIKELY
 * @brief Hint that the expression is likely to be true
 *
 * This macro tells the compiler that the expression is likely to evaluate
 * to true, allowing it to optimize the generated code for the common case.
 *
 * @param x The expression to evaluate
 * @return The same expression value
 */
#if defined(__clang__) || defined(__GNUC__)
#    define ELL_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#    define ELL_LIKELY(x) (!!(x))
#endif

/**
 * @def ELL_UNLIKELY
 * @brief Hint that the expression is unlikely to be true
 *
 * This macro tells the compiler that the expression is unlikely to evaluate
 * to true, allowing it to optimize the generated code for the uncommon case.
 * This is typically used for error conditions or exceptional paths.
 *
 * @param x The expression to evaluate
 * @return The same expression value
 */
#if defined(__clang__) || defined(__GNUC__)
#    define ELL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#    define ELL_UNLIKELY(x) (!!(x))
#endif
