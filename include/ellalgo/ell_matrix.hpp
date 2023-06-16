#pragma once

#include <valarray>

/** Square matrix */
class Matrix {
  size_t ndim;
  std::valarray<double> data;

public:
  /**
   * @brief Construct a new Matrix object
   *
   * @param ndim
   * @param init
   */
  explicit Matrix(size_t ndim, double init = 0.0)
      : ndim{ndim}, data(init, ndim * ndim) {}

  /**
   * @brief
   *
   * @param value
   */
  void clear(double value = 0.0) { data = value; }

  /**
   * @brief
   *
   * @param row
   * @param col
   * @return double&
   */
  double &operator()(size_t row, size_t col) {
    return this->data[row * ndim + col];
  }

  /**
   * @brief
   *
   * @param row
   * @param col
   * @return const double&
   */
  const double &operator()(size_t row, size_t col) const {
    return this->data[row * ndim + col];
  }

  /**
   * @brief
   *
   * @param alpha
   * @return Matrix&
   */
  Matrix &operator*=(double alpha) {
    this->data *= alpha;
    return *this;
  }

  /**
   * @brief
   *
   * @param alpha
   * @return Matrix
   */
  Matrix operator*(double alpha) const {
    Matrix res(*this);
    return res *= alpha;
  }

  /**
   * @brief
   *
   */
  void identity() {
    this->clear();
    this->diagonal() = 1;
  }

  /**
   * @brief
   *
   * @return std::slice_array<double>
   */
  std::slice_array<double> diagonal() {
    return this->data[std::slice(0, ndim, ndim + 1)];
  }

  /**
   * @brief
   *
   * @return std::slice_array<double>
   */
  std::slice_array<double> secondary_diagonal() {
    return this->data[std::slice(ndim - 1, ndim, ndim - 1)];
  }

  /**
   * @brief
   *
   * @param row
   * @return std::slice_array<double>
   */
  std::slice_array<double> row(std::size_t row) {
    return this->data[std::slice(ndim * row, ndim, 1)];
  }

  /**
   * @brief
   *
   * @param col
   * @return std::slice_array<double>
   */
  std::slice_array<double> column(std::size_t col) {
    return this->data[std::slice(col, ndim, ndim)];
  }

  /**
   * @brief
   *
   * @return double
   */
  double trace() const {
    return this->data[std::slice(0, ndim, ndim + 1)].sum();
  }
};
