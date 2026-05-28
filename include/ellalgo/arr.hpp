#pragma once

/// Minimal flat-vector array class replacing xtensor for small optimization problems.
/// Supports 1D (vector) and 2D (row-major matrix) with operations
/// needed by ellipsoid-method-based solvers. Not a general-purpose array library.

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
class Arr {
  public:
    using value_type = double;

    Arr() = default;

    explicit Arr(size_t n) : _data(n, 0.0), _rows(n) {}
    Arr(size_t r, size_t c) : _data(r * c, 0.0), _rows(r), _cols(c) {}
    Arr(size_t r, size_t c, double val) : _data(r * c, val), _rows(r), _cols(c) {}
    // Note: Arr(size_t, double) removed to avoid ambiguity with Arr(size_t, size_t).
    // Use Arr(n) / zeros(n) for zero-initialized 1D arrays.
    Arr(std::initializer_list<double> il) : _data(il), _rows(il.size()) {}
    explicit Arr(std::vector<double> v) : _data(std::move(v)), _rows(_data.size()) {}

    double& operator()(size_t i) { return _data[i]; }
    const double& operator()(size_t i) const { return _data[i]; }
    double& operator()(size_t i, size_t j) { return _data[i * _cols + j]; }
    const double& operator()(size_t i, size_t j) const { return _data[i * _cols + j]; }
    double& operator[](size_t i) { return _data[i]; }
    const double& operator[](size_t i) const { return _data[i]; }

    size_t size() const { return _data.size(); }
    size_t rows() const { return _rows; }
    size_t cols() const { return _cols; }
    bool is_2d() const { return _cols > 0; }
    double* data() { return _data.data(); }
    const double* data() const { return _data.data(); }
    auto begin() { return _data.begin(); }
    auto begin() const { return _data.begin(); }
    auto end() { return _data.end(); }
    auto end() const { return _data.end(); }

    Arr& operator+=(double s) {
        for (auto& v : _data) v += s;
        return *this;
    }
    Arr& operator-=(double s) {
        for (auto& v : _data) v -= s;
        return *this;
    }
    Arr& operator*=(double s) {
        for (auto& v : _data) v *= s;
        return *this;
    }
    Arr& operator+=(const Arr& other) {
        assert(_data.size() == other._data.size());
        for (size_t i = 0; i < _data.size(); ++i) _data[i] += other._data[i];
        return *this;
    }
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
struct Range {
    size_t start = 0;
    size_t end = 0;
    size_t step = 1;

    Range() = default;
    explicit Range(size_t end) : end(end) {}
    Range(size_t start, size_t end) : start(start), end(end) {}
    Range(size_t start, size_t end, size_t step) : start(start), end(end), step(step) {}
    static constexpr size_t ALL = SIZE_MAX;
};

inline constexpr size_t ALL = Range::ALL;

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------
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
inline Arr zeros(size_t n) {
    Arr a(n);
    return a;
}
inline Arr zeros(size_t r, size_t c) { return Arr(r, c); }
inline Arr ones(size_t r, size_t c) { return Arr(r, c, 1.0); }

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

inline Arr arange(double start, double end) {
    size_t n = (end > start) ? static_cast<size_t>(end - start) : 0;
    Arr out(n);
    for (size_t i = 0; i < n; ++i) out(i) = start + static_cast<double>(i);
    return out;
}

inline Arr make_same_shape(const Arr& a) {
    return a.is_2d() ? Arr(a.rows(), a.cols()) : Arr(a.rows());
}

// ---------------------------------------------------------------------------
// Element-wise math
// ---------------------------------------------------------------------------
inline Arr cos(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::cos(a(i));
    return o;
}
inline Arr log(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::log(a(i));
    return o;
}
inline Arr abs(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::abs(a(i));
    return o;
}
inline Arr exp(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::exp(a(i));
    return o;
}
inline Arr sqrt(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = std::sqrt(a(i));
    return o;
}

inline double sum(const Arr& a) { return std::accumulate(a.begin(), a.end(), 0.0); }

// ---------------------------------------------------------------------------
// where
// ---------------------------------------------------------------------------
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
inline Arr operator-(const Arr& a) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = -a(i);
    return o;
}
inline Arr operator+(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) + b(i);
    return o;
}
inline Arr operator-(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) - b(i);
    return o;
}
inline Arr operator*(const Arr& a, const Arr& b) {
    assert(a.size() == b.size());
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = a(i) * b(i);
    return o;
}
inline Arr operator*(double s, const Arr& a) { return Arr(a) *= s; }
inline Arr operator*(const Arr& a, double s) { return Arr(a) *= s; }
inline Arr operator/(const Arr& a, double s) { return Arr(a) *= (1.0 / s); }

// Comparison operators
inline Arr operator<=(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) <= s) ? 1.0 : 0.0;
    return o;
}
inline Arr operator>=(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) >= s) ? 1.0 : 0.0;
    return o;
}
inline Arr operator<(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) < s) ? 1.0 : 0.0;
    return o;
}
inline Arr operator>(const Arr& a, double s) {
    Arr o = make_same_shape(a);
    for (size_t i = 0; i < a.size(); ++i) o(i) = (a(i) > s) ? 1.0 : 0.0;
    return o;
}

// eval — identity for Arr
inline Arr eval(Arr a) { return a; }
