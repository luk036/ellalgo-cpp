/**
 * @file half_nonnegative.hpp
 * @brief Utility functions for computing half of non-negative numbers
 *
 * Copyright 2019 Denis Yaroshevskiy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ALGO_HALF_NONNEGATIVE_H
#define ALGO_HALF_NONNEGATIVE_H

#include <type_traits>
#include <utility>

namespace algo {

    /**
     * @brief Compute half of a non-negative integral number
     * 
     * This function computes half of a non-negative integral number.
     * For integral types, it uses unsigned arithmetic to avoid overflow
     * when dealing with negative numbers.
     * 
     * @tparam N The integral type
     * @param[in] n The non-negative number to halve
     * @return Half of the input number
     */
    template <typename N> auto half_nonnegative(N n) noexcept ->
        typename std::enable_if_t<std::is_integral<N>::value, N> {
        using UN = typename std::make_unsigned_t<N>;
        return static_cast<N>(static_cast<UN>(n) / 2);
    }

    /**
     * @brief Compute half of a non-negative floating-point number
     * 
     * This function computes half of a non-negative floating-point number.
     * For non-integral types, it simply divides by 2.
     * 
     * @tparam N The non-integral type
     * @param[in] n The non-negative number to halve
     * @return Half of the input number
     */
    template <typename N> auto half_nonnegative(N n) noexcept ->
        typename std::enable_if_t<!std::is_integral<N>::value, N> {
        return std::move(n) / 2;
    }

}  // namespace algo

#endif  // ALGO_HALF_NONNEGATIVE_H