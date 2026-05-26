#pragma once

#include <cstddef>
#include <valarray>
#include <vector>

/**
 * @brief Proxy view over a strided slice of a flat array
 *
 * Mimics the subset of std::slice_array<double> needed by the Matrix API:
 * element access, assignment from valarray, assignment from scalar, and
 * implicit conversion back to valarray.  Keeps all user code unchanged.
 */
class SliceView {
    double* const _data;
    const std::size_t _stride;
    const std::size_t _len;

  public:
    SliceView(double* data, std::size_t stride, std::size_t len) noexcept
        : _data{data}, _stride{stride}, _len{len} {}

    double& operator[](std::size_t i) { return this->_data[i * this->_stride]; }
    const double& operator[](std::size_t i) const { return this->_data[i * this->_stride]; }

    SliceView& operator=(const std::valarray<double>& v) {
        for (std::size_t i = 0; i < this->_len; ++i) this->_data[i * this->_stride] = v[i];
        return *this;
    }

    SliceView& operator=(double scalar) {
        for (std::size_t i = 0; i < this->_len; ++i) this->_data[i * this->_stride] = scalar;
        return *this;
    }

    operator std::valarray<double>() const {
        std::valarray<double> res(this->_len);
        for (std::size_t i = 0; i < this->_len; ++i) res[i] = this->_data[i * this->_stride];
        return res;
    }

    double sum() const {
        double s = 0.0;
        for (std::size_t i = 0; i < this->_len; ++i) s += this->_data[i * this->_stride];
        return s;
    }
};

/**
 * @brief Square matrix with flat std::vector<double> storage
 *
 * Provides element access via operator()(row, col) and strided views
 * (row, column, diagonal) via SliceView, which supports assignment
 * from valarray, scalar fill, element access, and implicit valarray
 * conversion — matching the original std::slice_array<double> API.
 */
class Matrix {
    std::size_t ndim;
    std::vector<double> data;

  public:
    /**
     * @brief Construct an ndim × ndim matrix
     * @param[in] ndim  Number of rows/columns
     * @param[in] init  Fill value (default 0.0)
     */
    explicit Matrix(std::size_t ndim, double init = 0.0) : ndim{ndim}, data(ndim * ndim, init) {}

    /** @brief Reset all elements to value */
    void clear(double value = 0.0) { std::fill(this->data.begin(), this->data.end(), value); }

    /** @brief Access element (row, col) — mutable */
    double& operator()(std::size_t row, std::size_t col) {
        return this->data[row * this->ndim + col];
    }

    /** @brief Access element (row, col) — const */
    const double& operator()(std::size_t row, std::size_t col) const {
        return this->data[row * this->ndim + col];
    }

    /** @brief Scale all elements by alpha */
    Matrix& operator*=(double alpha) {
        for (auto& v : this->data) v *= alpha;
        return *this;
    }

    /** @brief Return a scaled copy */
    Matrix operator*(double alpha) const {
        Matrix res(*this);
        return res *= alpha;
    }

    /** @brief Set to identity matrix */
    void identity() {
        this->clear();
        auto d = this->diagonal();
        for (std::size_t i = 0; i < this->ndim; ++i) d[i] = 1.0;
    }

    /** @brief Mutable view of the diagonal (stride = ndim + 1) */
    SliceView diagonal() { return SliceView(this->data.data(), this->ndim + 1, this->ndim); }

    /** @brief Mutable view of row r (contiguous, stride = 1) */
    SliceView row(std::size_t r) {
        return SliceView(this->data.data() + r * this->ndim, 1, this->ndim);
    }

    /** @brief Mutable view of column c (strided, stride = ndim) */
    SliceView column(std::size_t c) {
        return SliceView(this->data.data() + c, this->ndim, this->ndim);
    }

    /** @brief Sum of diagonal elements */
    double trace() const { return const_cast<Matrix*>(this)->diagonal().sum(); }
};
