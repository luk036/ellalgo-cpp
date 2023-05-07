#pragma once

#include <valarray>

/** Square matrix */
class Matrix {
  size_t ndim;
  std::valarray<double> data;

public:
  explicit Matrix(size_t ndim, double init = 0.0)
      : ndim{ndim}, data(init, ndim * ndim) {}

  void clear(double value = 0.0) { data = value; }

  double &operator()(size_t r, size_t c) { return this->data[r * ndim + c]; }

  const double &operator()(size_t r, size_t c) const {
    return this->data[r * ndim + c];
  }

  Matrix &operator*=(double alpha) {
    this->data *= alpha;
    return *this;
  }

  Matrix operator*(double alpha) const {
    Matrix res(*this);
    return res *= alpha;
  }

  void identity() {
    this->clear();
    this->diagonal() = 1;
  }

  std::slice_array<double> diagonal() {
    return this->data[std::slice(0, ndim, ndim + 1)];
  }

  std::slice_array<double> secondary_diagonal() {
    return this->data[std::slice(ndim - 1, ndim, ndim - 1)];
  }

  std::slice_array<double> row(std::size_t row) {
    return this->data[std::slice(ndim * row, ndim, 1)];
  }

  std::slice_array<double> column(std::size_t col) {
    return this->data[std::slice(col, ndim, ndim)];
  }

  double trace() const {
    return this->data[std::slice(0, ndim, ndim + 1)].sum();
  }
};
