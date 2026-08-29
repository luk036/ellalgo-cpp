/**
 * @file lmi_problem.hpp
 * @brief LMI problem facade: owns data and drives the cutting-plane loop
 */

#pragma once

#include <tuple>    // for tuple
#include <utility>  // for move
#include <valarray>
#include <vector>

#include "cutting_plane.hpp"
#include "ell_stable.hpp"
#include "oracles/lmi_oracle.hpp"

/**
 * @brief LMI feasibility problem facade.
 *
 * Owns the LMI data (F matrices and constant term B) and the lazily-created
 * LmiOracle, then drives the cutting-plane method through the standard
 * `cutting_plane_feas` driver. Hides the 3-step recipe
 * (build oracle -> build search space -> call driver) behind a single call.
 *
 * The LMI feasibility problem is:
 *
 *     find  x
 *     s.t.  (B - F * x) >= 0   (i.e. B - Σ F_k x_k is positive semidefinite)
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 */
template <typename Arr036, typename Mat = Arr036> class LMIProblem {
    using Vec = std::valarray<double>;

    size_t _ndim;
    std::vector<Mat> _F;      // problem data, must precede _omega
    Mat _B;                   // constant term
    LmiOracle<Arr036, Mat> _omega;  // holds a reference to _F

  public:
    /**
     * @brief Construct a new LMIProblem object
     *
     * @param[in] ndim Dimension of the decision space
     * @param[in] F    Vector of matrices F_i (moved in)
     * @param[in] B    Constant term (moved in)
     */
    LMIProblem(size_t ndim, std::vector<Mat> F, Mat B)
        : _ndim{ndim}, _F{std::move(F)}, _B{std::move(B)}, _omega{ndim, _F, _B} {}

    /**
     * @brief Solve the LMI feasibility problem.
     *
     * Builds an EllStable search space with the given per-axis radii and
     * initial center, then runs the cutting-plane feasibility method.
     *
     * @param[in] radii  Per-axis radii of the initial ellipsoid
     * @param[in] xc     Initial center point (moved in)
     * @param[in] options Maximum iteration and error tolerance etc.
     * @return Tuple (solution x, number of iterations)
     */
    auto solve_feas(const Vec& radii, Arr036 xc, const Options& options = Options())
        -> std::tuple<Arr036, size_t> {
        EllStable<Arr036> space{radii, std::move(xc)};
        return cutting_plane_feas(this->_omega, space, options);
    }

    /**
     * @brief Solve the LMI feasibility problem (alpha-scaled initial space).
     *
     * @param[in] alpha Scaling factor for the initial ellipsoid
     * @param[in] xc    Initial center point (moved in)
     * @param[in] options Maximum iteration and error tolerance etc.
     * @return Tuple (solution x, number of iterations)
     */
    auto solve_feas(double alpha, Arr036 xc, const Options& options = Options())
        -> std::tuple<Arr036, size_t> {
        EllStable<Arr036> space{alpha, std::move(xc)};
        return cutting_plane_feas(this->_omega, space, options);
    }
};

/**
 * @brief Create an LMIProblem facade.
 *
 * @tparam Arr036 Array type for the decision variables
 * @tparam Mat    Matrix type (defaults to Arr036)
 * @param[in] ndim Dimension of the decision space
 * @param[in] F    Vector of matrices F_i (moved in)
 * @param[in] B    Constant term (moved in)
 * @return LMIProblem<Arr036, Mat>
 */
template <typename Arr036, typename Mat = Arr036>
inline auto make_lmi_problem(size_t ndim, std::vector<Mat> F, Mat B) -> LMIProblem<Arr036, Mat> {
    return {ndim, std::move(F), std::move(B)};
}
