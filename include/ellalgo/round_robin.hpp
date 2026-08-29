/**
 * @file round_robin.hpp
 * @brief Round-robin index helper for cyclic constraint scanning
 */

#pragma once

#include <cstddef>

/**
 * @brief Round-robin index generator over a half-open range [lo, hi)
 *
 * Successive calls to next() yield lo, lo+1, ..., hi-1, lo, ...
 * Used to scan constraints cyclically (e.g. ProfitOracle, LowpassOracle).
 * The first call to next() returns `lo`, matching the `++idx; if (idx == N)
 * idx = 0;` idiom it replaces.
 */
class RoundRobin {
  public:
    /// @brief Default-construct (uninitialized; reassign before use)
    RoundRobin() : _lo{0}, _hi{0}, _cur{0} {}

    /// @brief Round-robin over [0, hi)
    explicit RoundRobin(std::size_t hi) : RoundRobin{0, hi} {}

    /// @brief Round-robin over [lo, hi)
    RoundRobin(std::size_t lo, std::size_t hi) : _lo{lo}, _hi{hi}, _cur{hi - 1} {}

    /// @brief Advance to the next index and return it
    auto next() -> std::size_t {
        if (++this->_cur == this->_hi) {
            this->_cur = this->_lo;
        }
        return this->_cur;
    }

  private:
    std::size_t _lo;
    std::size_t _hi;
    std::size_t _cur;
};
