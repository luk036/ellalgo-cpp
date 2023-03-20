#pragma once

#include <valarray>

/** Square matrix */
class Matrix {
  int dim;
  std::valarray<double> data;

public:
  explicit Matrix(int dim, double init = 0.0)
      : dim{dim}, data(init, dim * dim) {}

  void clear(double value = 0.0) { data = value; }

  double &operator()(int r, int c) { return this->data[r * dim + c]; }

  const double &operator()(int r, int c) const {
    return this->data[r * dim + c];
  }

  Matrix &operator*=(double alpha) {
    this->data *= alpha;
    return *this;
  }

  void identity() {
    this->clear();
    this->diagonal() = 1;
  }

  std::slice_array<double> diagonal() {
    return this->data[std::slice(0, dim, dim + 1)];
  }

  std::slice_array<double> secondary_diagonal() {
    return this->data[std::slice(dim - 1, dim, dim - 1)];
  }

  std::slice_array<double> row(std::size_t row) {
    return this->data[std::slice(dim * row, dim, 1)];
  }

  std::slice_array<double> column(std::size_t col) {
    return this->data[std::slice(col, dim, dim)];
  }

  double trace() const { return this->data[std::slice(0, dim, dim + 1)].sum(); }
};
