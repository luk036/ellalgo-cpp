/**
 * @file main_core.cpp
 * @brief Consumer test for the EllAlgo::Core component (cutting-plane framework)
 *
 * Links ONLY EllAlgo::Core — proving the framework (concepts + drivers) is
 * dependency-free (no spdlog/fmt, no compiled library). The consumer supplies
 * its own SearchSpace and oracle (Strategy pattern instances), exactly like
 * netoptim/corr-solver/multiplierless do.
 */

#include <cmath>
#include <ellalgo/cutting_plane.hpp>
#include <iostream>
#include <optional>
#include <utility>

// Minimal 1-D search space satisfying the SearchSpace concept.
class IntervalSpace {
  public:
    using ArrayType = double;

    explicit IntervalSpace(const double l, const double u) noexcept
        : _r{(u - l) / 2}, _xc{l + _r} {}

    auto xc() const noexcept -> double { return _xc; }
    auto tsq() const noexcept -> double { return _tsq; }

    auto update_bias_cut(const std::pair<double, double>& cut) noexcept -> CutStatus {
        const auto& g = cut.first;
        const auto& beta = cut.second;
        const auto tau = std::abs(_r * g);
        _tsq = tau * tau;
        if (beta > tau) {
            return CutStatus::NoSoln;
        }
        if (beta < -tau) {
            return CutStatus::NoEffect;
        }
        const auto bound = _xc - beta / g;
        const auto u = g > 0.0 ? bound : _xc + _r;
        const auto l = g > 0.0 ? _xc - _r : bound;
        _r = (u - l) / 2;
        _xc = l + _r;
        return CutStatus::Success;
    }

    auto update_central_cut(const std::pair<double, double>& cut) noexcept -> CutStatus {
        const auto& g = cut.first;
        const auto tau = std::abs(_r * g);
        _tsq = tau * tau;
        _r /= 2;
        _xc += g > 0.0 ? -_r : _r;
        return CutStatus::Success;
    }

  private:
    double _r;
    double _xc;
    double _tsq = 0.0;
};

// Feasibility oracle: find a point with x >= 3 (cut: 3 - x <= 0).
class ThresholdOracle {
  public:
    auto assess_feas(const double x) -> std::optional<std::pair<double, double>> {
        if (x >= 3.0) {
            return std::nullopt;  // feasible
        }
        return std::make_pair(-1.0, 3.0 - x);
    }
};

auto main() -> int {
    auto omega = ThresholdOracle{};
    auto space = IntervalSpace{0.0, 4.0};
    const auto options = Options{1000, 1e-10};
    const auto [x, niter] = cutting_plane_feas(omega, space, options);
    std::cout << "ellalgo-cpp Core component test: x = " << x << " after " << niter
              << " iterations\n";
    return x >= 3.0 ? 0 : 1;
}
