// -*- coding: utf-8 -*-
#pragma once

#include "../ell_matrix.hpp"
#include <cassert> // for assert
#include <cstddef> // for size_t
#include <utility> // for pair
#include <valarray>

/**
 * @brief LDLT factorization for LMI
 *
 *  - LDL^T square-root-free version
 *  - Option allow semidefinite
 *  - A matrix A in R^{m x m} is positive definite iff v' A v > 0
 *      for all v in R^n.
 *  - O(p^2) per iteration, independent of N
 */
class LDLTMgr {
  using Vec = std::valarray<double>;
  using Rng = std::pair<size_t, size_t>;

public:
  Rng p{0U, 0U};   //!< the rows where the process starts and stops
  Vec witness_vec; //!< witness vector
  const size_t _n; //!< dimension

private:
  Matrix T; //!< temporary storage

  // static Vec zeros_vec(size_t n);
  // static Mat zeros_mat(size_t n);

public:
  /**
   * @brief Construct a new ldlt ext object
   *
   * @param[in] N dimension
   */
  explicit LDLTMgr(size_t N) : witness_vec(0.0, N), _n{N}, T{N} {}

  LDLTMgr(const LDLTMgr &) = delete;
  LDLTMgr &operator=(const LDLTMgr &) = delete;
  LDLTMgr(LDLTMgr &&) = default;

  /**
   * @brief Perform LDLT Factorization
   *
   * @param[in] A Symmetric Matrix
   *
   * If $A$ is positive definite, then $p$ is zero.
   * If it is not, then $p$ is a positive integer,
   * such that $v = R^-1 e_p$ is a certificate vector
   * to make $v'*A[:p,:p]*v < 0$
   */
  // auto factorize(const Arr036 &A) -> bool {
  //   return this->factor([&](size_t i, size_t j) { return A(i, j); });
  // }

  /**
   * @brief Perform LDLT Factorization
   *
   * @param[in] A Symmetric Matrix
   *
   * If $A$ is positive definite, then $p$ is zero.
   * If it is not, then $p$ is a positive integer,
   * such that $v = R^-1 e_p$ is a certificate vector
   * to make $v'*A[:p,:p]*v < 0$
   */
  template <typename Mat> auto factorize(const Mat &A) -> bool {
    return this->factor([&](size_t i, size_t j) { return A(i, j); });
  }

  /**
   * @brief Perform LDLT Factorization (Lazy evaluation)
   *
   * @tparam Fn
   * @param[in] get_matrix_elem function to access the elements of A
   *
   * See also: factorize()
   */
  template <typename Callable> auto factor(Callable &&get_matrix_elem) -> bool {
    this->p = {0U, 0U};
    // auto &[start, stop] = this->p;
    auto &start = this->p.first;
    auto &stop = this->p.second;

    for (auto i = 0U; i != this->_n; ++i) {
      // auto j = start;
      auto d = get_matrix_elem(i, start);
      for (auto j = start; j != i; ++j) {
        this->T(j, i) = d;
        this->T(i, j) = d / this->T(j, j); // note: T(j, i) here!
        auto s = j + 1;
        d = get_matrix_elem(i, s);
        for (auto k = start; k != s; ++k) {
          d -= this->T(i, k) * this->T(k, s);
        }
      }
      this->T(i, i) = d;

      if (d <= 0.0) {
        stop = i + 1;
        break;
      }
    }

    return this->is_spd();
  }

  /**
   * @brief Perform LDLT Factorization (Lazy evaluation)
   *
   * @tparam Fn
   * @param[in] get_matrix_elem function to access the elements of A
   *
   * See also: factorize()
   */
  template <typename Callable>
  auto factor_with_allow_semidefinte(Callable &&get_matrix_elem) -> bool {
    this->p = {0U, 0U};
    // auto &[start, stop] = this->p;
    auto &start = this->p.first;
    auto &stop = this->p.second;

    for (auto i = 0U; i != this->_n; ++i) {
      // auto j = start;
      auto d = get_matrix_elem(i, start);
      for (auto j = start; j != i; ++j) {
        this->T(j, i) = d;
        this->T(i, j) = d / this->T(j, j); // note: T(j, i) here!
        auto s = j + 1;
        d = get_matrix_elem(i, s);
        for (auto k = start; k != s; ++k) {
          d -= this->T(i, k) * this->T(k, s);
        }
      }
      this->T(i, i) = d;

      if (d < 0.0) {
        // this->stop = i + 1;
        stop = i + 1;
        break;
      }
      if (d == 0.0) {
        start = i + 1;
        // restart at i + 1, special as an LMI oracle
      }
    }
    return this->is_spd();
  }

  /**
   * @brief Is $A$ symmetric positive definite (spd)
   *
   * @return true
   * @return false
   */
  auto is_spd() const noexcept -> bool { return this->p.second == 0; }

  /**
   * @brief witness that certifies $A$ is not
   * symmetric positive definite (spd)
   *
   * @return auto
   */
  auto witness() -> double {
    assert(!this->is_spd());

    // const auto& [start, n] = this->p;
    const auto &start = this->p.first;
    const auto &n = this->p.second;
    auto m = n - 1; // assume stop > 0
    this->witness_vec[m] = 1.0;
    for (auto i = m; i > start; --i) {
      this->witness_vec[i - 1] = 0.0;
      for (auto k = i; k != n; ++k) {
        this->witness_vec[i - 1] -= this->T(k, i - 1) * this->witness_vec[k];
      }
    }
    return -this->T(m, m);
  }

  template <typename Arr036> auto set_witness_vec(Arr036 &v) const -> void {
    for (auto i = 0U; i != this->_n; ++i) {
      v[i] = this->witness_vec[i];
    }
  }

  /**
   * @brief Calculate v'*{A}(p,p)*v
   *
   * @param[in] A
   * @return double
   */
  template <typename Mat> auto sym_quad(const Mat &A) const -> double {
    auto res = double{};
    const auto &v = this->witness_vec;
    // const auto& [start, stop] = this->p;
    const auto &start = this->p.first;
    const auto &stop = this->p.second;
    for (auto i = start; i != stop; ++i) {
      auto s = 0.0;
      for (auto j = i + 1; j != stop; ++j) {
        s += A(i, j) * v[j];
      }
      res += v[i] * (A(i, i) * v[i] + 2.0 * s);
    }
    return res;
  }

  /**
   * @brief Return upper triangular matrix $R$ where $A = R^T R$
   *
   * Note: must input a zero matrix
   * @return typename LDLTMgr<Arr036>::Mat
   */
  template <typename Mat> auto sqrt(Mat &M) -> void {
    assert(this->is_spd());

    for (auto i = 0U; i != this->_n; ++i) {
      M(i, i) = std::sqrt(this->T(i, i));
      for (auto j = i + 1; j != this->_n; ++j) {
        M(i, j) = this->T(j, i) * M(i, i);
        M(j, i) = 0.0;
      }
    }
  }
};
