// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <optional>
#include <vector>

struct MyOracle {
    using ArrayType = std::vector<double>;
    using Cut = std::pair<ArrayType, double>;

    auto assess_feas(const ArrayType& x) -> std::optional<Cut> {
        double f = x[0] + x[1] - 3.0;
        if (f > 0.0) {
            return std::make_optional<Cut>({{1.0, 1.0}, f});
        }
        return {};
    }
};

TEST_CASE("Cutting-plane feasibility") {
    auto ell = Ell<std::vector<double>>(10.0, std::vector<double>{0.0, 0.0});
    auto oracle = MyOracle{};
    auto options = Options();
    options.max_iters = 2000;
    options.tolerance = 1e-8;

    auto result = cutting_plane_feas(oracle, ell, options);
    CHECK(std::get<0>(result).size() != 0);
}

struct MyOracle2 {
    using ArrayType = std::vector<double>;
    using Cut = std::pair<ArrayType, double>;

    auto assess_optim(const ArrayType& x, double& t) -> std::pair<Cut, bool> {
        double f = x[0] + x[1];
        if (f > t) {
            return {{{1.0, 1.0}, f - t}, false};
        } else {
            t = f;
            return {{{-1.0, -1.0}, 0.0}, true};
        }
    }
};

TEST_CASE("Cutting-plane optimization") {
    auto ell = Ell<std::vector<double>>(10.0, std::vector<double>{0.0, 0.0});
    auto oracle = MyOracle2{};
    auto options = Options();
    options.max_iters = 2000;
    options.tolerance = 1e-8;
    double t = 1e100;

    auto result = cutting_plane_optim(oracle, ell, t, options);
    CHECK(std::get<0>(result).size() != 0);
}

struct MyOracle3 {
    using ArrayType = std::vector<double>;
    using Cut = std::pair<ArrayType, double>;

    double t = 0.0;

    void update(double t) {
        this->t = t;
    }

    auto assess_feas(const ArrayType& x) -> std::optional<Cut> {
        double f = x[0] + x[1] - 3.0;
        if (f > this->t) {
            return std::make_optional<Cut>({{1.0, 1.0}, f - this->t});
        }
        return {};
    }
};

TEST_CASE("Binary search") {
    auto ell = Ell<std::vector<double>>(10.0, std::vector<double>{0.0, 0.0});
    auto oracle = MyOracle3{};
    auto options = Options();
    options.max_iters = 2000;
    options.tolerance = 1e-8;

    auto adaptor = BSearchAdaptor(oracle, ell, options);
    auto result = bsearch(adaptor, std::make_pair(0.0, 10.0), options);
    CHECK(std::get<0>(result) < 3.0);
}
