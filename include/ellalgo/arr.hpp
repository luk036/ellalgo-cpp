/**
 * @file arr.hpp
 * @brief Minimal flat-vector array class replacing xtensor for small optimization problems.
 *
 * Supports 1D (vector) and 2D (row-major matrix) with operations
 * needed by ellipsoid-method-based solvers. Not a general-purpose array library.
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>  // for SIZE_MAX
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Arr — 1D or 2D array backed by std::vector<double>
// ---------------------------------------------------------------------------
/// @brief 1D or 2D array backed by std::vector<double> for small optimization problems
class Arr {
  public:
    using value_type = double;

    /// @brief Default constructor (empty array)
    Arr() = default;

    /// @brief Construct 1D array of size n, zero-initialized
    explicit Arr(size_t n) : _data(n, 0.0), _rows(n) {}
    /// @brief Construct 2D array with r rows, c columns, zero-initialized
    Arr(size_t r, size_t c) : _data(r * c, 0.0), _rows(r), _cols(c) {}
    /// @brief Construct 2D array with r rows, c columns, fill value val
    Arr(size_t r, size_t c, double val) : _data(r * c, val), _rows(r), _cols(c) {}
    // Note: Arr(size_t, double) removed to avoid ambiguity with Arr(size_t, size_t).
    // Use Arr(n) / zeros(n) for zero-initialized 1D arrays.
    /// @brief Construct from initializer list (1D)
    Arr(std::initializer_list<double> il) : _data(il), _rows(il.size()) {}
    /// @brief Construct from vector (1D)
    explicit Arr(std::vector<double> v) : _data(std::move(v)), _rows(_data.size()) {}

    /// @brief 1D element access (mutable)
    double& operator()(size_t i) { return _data[i]; }
    /// @brief 1D element access (const)
    const double& operator()(size_t i) const { return _data[i]; }
    /// @brief 2D element access (mutable), row-major
    double& operator()(size_t i, size_t j) { return _data[i * _cols + j]; }
    /// @brief 2D element access (const), row-major
    const double& operator()(size_t i, size_t j) const { return _data[i * _cols + j]; }
    /// @brief 1D element access (mutable)
    double& operator[](size_t i) { return _data[i]; }
    /// @brief 1D element access (const)
    const double& operator[](size_t i) const { return _data[i]; }

    /// @brief Total number of elements
    size_t size() const { return _data.size(); }
    /// @brief Number of rows (1 for 1D arrays)
    size_t rows() const { return _rows; }
    /// @brief Number of columns (0 for 1D arrays)
    size_t cols() const { return _cols; }
    /// @brief Whether this is a 2D array (cols > 0)
    bool is_2d() const { return _cols > 0; }
    /// @brief Raw pointer to underlying data (mutable)
    double* data() { return _data.data(); }
    /// @brief Raw pointer to underlying data (const)
    const double* data() const { return _data.data(); }
    /// @brief Iterator to beginning (mutable)
    auto begin() { return _data.begin(); }
    /// @brief Iterator to beginning (const)
    auto begin() const { return _data.begin(); }
    /// @brief Iterator past the end (mutable)
    auto end() { return _data.end(); }
    /// @brief Iterator past the end (const)
    auto end() const { return _data.end(); }

    /// @brief Add scalar to all elements
    Arr& operator+=(double s) {
        for (auto& v : _data) v += s;
        return *this;
    }
    /// @brief Subtract scalar from all elements
    Arr& operator-=(double s) {
        for (auto& v : _data) v -= s;
        return *this;
    }
    /// @brief Multiply all elements by scalar
    Arr& operator*=(double s) {
        for (auto& v : _data) v *= s;
        return *this;
    }
    /// @brief Element-wise addition with another Arr
    Arr& operator+=(const Arr& other) {
        assert(_data.size() == other._data.size());
        for (size_t i = 0; i < _data.size(); ++i) _data[i] += other._data[i];
        return *this;
    }
    /// @brief Element-wise subtraction with another Arr
    Arr& operator-=(const Arr& other) {
        assert(_data.size() == other._data.size());
        for (size_t i = 0; i < _data.size(); ++i) _data[i] -= other._data[i];
        return *this;
    }

  private:
    std::vector<double> _data;
    size_t _rows = 0;
    size_t _cols = 0;
};

// ---------------------------------------------------------------------------
// Range helper
// ---------------------------------------------------------------------------
/// @brief Index range [start, end) with step for slicing views
struct Range {
    size_t start = 0;  ///< Start index (inclusive)
    size_t end = 0;    ///< End index (exclusive)
    size_t step = 1;   ///< Step size

    /// @brief Default range (empty)
    Range() = default;
    /// @brief Range from 0 to end
    explicit Range(size_t end) : end(end) {}
    /// @brief Range [start, end)
    Range(size_t start, size_t end) : start(start), end(end) {}
    /// @brief Range [start, end) with given step
    Range(size_t start, size_t end, size_t step) : start(start), end(end), step(step) {}
    /// @brief Sentinel value meaning "all elements"
    static constexpr size_t ALL = SIZE_MAX;
};

/// @brief Convenience alias for Range::ALL
inline constexpr size_t ALL = Range::ALL;

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------
/// @brief Extract a submatrix view from a 2D array using Range for rows and cols
/// @param[in] a     Input 2D array
/// @param[in] rows  Row range [start, end, step]
/// @param[in] cols  Column range [start, end, step]
/// @return New Arr containing the submatrix
inline Arr view(const Arr& a, const Range& rows, const Range& cols) {
    assert(a.is_2d());
    auto rs = rows.start;
    auto re = (rows.end == Range::ALL) ? a.rows() : rows.end;
    auto cs = cols.start;
    auto ce = (cols.end == Range::ALL) ? a.cols() : cols.end;
    auto rstep = rows.step;
    auto cstep = cols.step;
    size_t out_r = (re - rs + rstep - 1) / rstep;
    size_t out_c = (ce - cs + cstep - 1) / cstep;
    Arr out(out_r, out_c);
    for (size_t i = 0; i < out_r; ++i)
        for (size_t j = 0; j < out_c; ++j) out(i, j) = a(rs + i * rstep, cs + j * cstep);
    return out;
}

/// @brief Extract a subarray view (1D or 2D) using a single Range
/// @param[in] a    Input array (1D or 2D)
/// @param[in] rows Row range (or element range for 1D)
/// @return New Arr containing the subarray
inline Arr view(const Arr& a, const Range& rows) {
    if (!a.is_2d()) {
        auto s = rows.start;
        auto e = (rows.end == Range::ALL) ? a.size() : rows.end;
        auto step = rows.step;
        size_t n = (e - s + step - 1) / step;
        Arr out(n);
        for (size_t i = 0; i < n; ++i) out(i) = a(s + i * step);
        return out;
    }
    return view(a, rows, Range(Range::ALL));
}

// ---------------------------------------------------------------------------
// Builder functions
// ---------------------------------------------------------------------------
/// @brief Create zero-initialized 1D array of size n
inline Arr zeros(size_t n) {
    Arr a(n);
    return a;
}
/// @brief Create zero-initialized 2D array with r rows, c columns
inline Arr zeros(size_t r, size_t c) { return Arr(r, c); }
/// @brief Create 2D array of ones with r rows, c columns
inline Arr ones(size_t r, size_t c) { return Arr(r, c, 1.0); }

/// @brief Linearly spaced values from start to end, inclusive
///
/// @f[
///     x_i = x_{\text{start}} + i \cdot \frac{x_{\text{end}} - x_{\text{start}}}{n - 1}, \qquad i = 0, \dots, n-1
/// @f]
/// @param[in] start First value
/// @param[in] end   Last value
/// @param[in] n     Number of points
/// @return 1D Arr of n equally spaced values
inline Arr linspace(double start, double end, size_t n) {
    Arr out(n);
    if (n == 0) return out;
    if (n == 1) {
        out(0) = start;
        return out;
    }
    double step = (end - start) / static_cast<double>(n - 1);
    for (size_t i = 0; i < n; ++i) out(i) = start + step * static_cast<double>(i);
    return out;
}

/// @brief Values from start to end-1 with step 1
/// @return 1D Arr containing [start, start+1, ..., end-1]
inline Arr arange(double start, double end) {
    size_t n = (end > start) ? static_cast<size_t>(end - start) : 0;
    Arr out(n);
    for (size_t i = 0; i < n; ++i) out(i) = start + static_cast<double>(i);
    return out;
}

/// @brief Create a zero-initialized Arr with the same shape as input
inline Arr make_same_shape(const Arr& a) {
    return a.is_2d() ? Arr(a.rows(), a.cols()) : Arr(a.rows());
}

// ---------------------------------------------------------------------------
// Element-wise math
// ---------------------------------------------------------------------------
/// @brief Element-wise cosine
inline Arr cos(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::cos(a(i));
    return o;
}
/// @brief Element-wise natural logarithm
inline Arr log(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::log(a(i));
    return o;
}
/// @brief Element-wise absolute value
inline Arr abs(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::abs(a(i));
    return o;
}
/// @brief Element-wise exponential
inline Arr exp(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::exp(a(i));
    return o;
}
/// @brief Element-wise square root
inline Arr sqrt(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::sqrt(a(i));
    return o;
}

/// @brief Sum of all elements
inline double sum(const Arr& a) { return std::accumulate(a.begin(), a.end(), 0.0); }

// ---------------------------------------------------------------------------
// where
// ---------------------------------------------------------------------------
/// @brief Find indices of non-zero elements
/// @param[in] condition Input 1D array (non-zero = true)
/// @return Vector containing [indices] as a 1D Arr
inline std::vector<Arr> where(const Arr& condition) {
    assert(!condition.is_2d());
    std::vector<size_t> idx;
    for (size_t i = 0; i < condition.size(); ++i)
        if (condition(i) != 0.0) idx.push_back(i);
    Arr indices(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) indices(i) = static_cast<double>(idx[i]);
    return {indices};
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------
/// @brief Matrix-vector multiplication
///
/// @f[
///     y = A x, \qquad y_i = \sum_{j=1}^{n} A_{ij} x_j
/// @f]
/// @param[in] A 2D matrix (m × n)
/// @param[in] x 1D vector (n)
/// @return Result vector (m)
inline Arr dot(const Arr& A, const Arr& x) {
    assert(A.is_2d() && !x.is_2d() && A.cols() == x.size());
    Arr out(A.rows());
    for (size_t i = 0; i < A.rows(); ++i) {
        double s = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) s += A(i, j) * x(j);
        out(i) = s;
    }
    return out;
}

/// @brief Outer product of two 1D vectors
///
/// @f[
///     C = u \otimes v, \qquad C_{ij} = u_i v_j
/// @f]
/// @param[in] u First 1D vector
/// @param[in] v Second 1D vector
/// @return 2D matrix where result(i,j) = u(i) × v(j)
inline Arr outer(const Arr& u, const Arr& v) {
    assert(!u.is_2d() && !v.is_2d());
    Arr out(u.size(), v.size());
    for (size_t i = 0; i < u.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j) out(i, j) = u(i) * v(j);
    return out;
}

// ---------------------------------------------------------------------------
// concatenate (axis=1 only)
// ---------------------------------------------------------------------------
/// @brief Concatenate two 2D arrays along columns (axis=1)
/// @param[in] a Left 2D array
/// @param[in] b Right 2D array
/// @return 2D array with rows = a.rows(), cols = a.cols() + b.cols()
inline Arr concatenate(const Arr& a, const Arr& b, int /* axis */ = 1) {
    assert(a.is_2d() && b.is_2d() && a.rows() == b.rows());
    size_t m = a.rows();
    size_t ca = a.cols();
    size_t cb = b.cols();
    Arr out(m, ca + cb);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < ca; ++j) out(i, j) = a(i, j);
        for (size_t j = 0; j < cb; ++j) out(i, ca + j) = b(i, j);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------
/// @brief Unary negation (element-wise)
inline Arr operator-(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = -a(i);
    return o;
}
/// @brief Element-wise addition of two arrays
inline Arr operator+(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) + b(i);
    return o;
}
/// @brief Element-wise subtraction of two arrays
inline Arr operator-(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) - b(i);
    return o;
}
/// @brief Element-wise multiplication of two arrays
inline Arr operator*(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) * b(i);
    return o;
}
/// @brief Scalar multiplication (scalar × array)
inline Arr operator*(double s, const Arr& a) { return Arr(a) *= s; }
/// @brief Scalar multiplication (array × scalar)
inline Arr operator*(const Arr& a, double s) { return Arr(a) *= s; }
/// @brief Scalar division (array / scalar)
inline Arr operator/(const Arr& a, double s) { return Arr(a) *= (1.0 / s); }

/// @brief Element-wise less-than-or-equal-to comparison with scalar
/// @return Arr of 1.0/0.0 indicating true/false
inline Arr operator<=(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) <= s) ? 1.0 : 0.0;
    return o;
}
/// @brief Element-wise greater-than-or-equal-to comparison with scalar
/// @return Arr of 1.0/0.0 indicating true/false
inline Arr operator>=(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) >= s) ? 1.0 : 0.0;
    return o;
}
/// @brief Element-wise less-than comparison with scalar
/// @return Arr of 1.0/0.0 indicating true/false
inline Arr operator<(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) < s) ? 1.0 : 0.0;
    return o;
}
/// @brief Element-wise greater-than comparison with scalar
/// @return Arr of 1.0/0.0 indicating true/false
inline Arr operator>(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) > s) ? 1.0 : 0.0;
    return o;
}

/// @brief Identity function for Arr (for API compatibility)
inline Arr eval(Arr a) { return a; }
