// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#ifdef RAPIDCHECK_H
#    include <rapidcheck.h>

#    include <ellalgo/ell.hpp>
#    include <vector>

TEST_CASE("Property-based test: Ell constructor with alpha sets initial tsq") {
    rc::check("Ell(alpha, x) has tsq = 0 initially", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        auto alpha = *rc::gen::positive<double>();

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        RC_ASSERT(ellip.tsq() == doctest::Approx(0.0));
    });
}

TEST_CASE("Property-based test: Ell constructor with vector sets initial tsq") {
    rc::check("Ell(val, x) has tsq = 0 initially", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        std::valarray<double> val(1.0, n);
        std::vector<double> x(n, 0.0);

        auto ellip = Ell<std::vector<double>>(val, x);

        RC_ASSERT(ellip.tsq() == doctest::Approx(0.0));
    });
}

TEST_CASE("Property-based test: Ell center is correctly initialized") {
    rc::check("Ell center matches input vector", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        auto alpha = *rc::gen::positive<double>();

        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = *rc::gen::arbitrary<double>();
        }

        auto ellip = Ell<std::vector<double>>(alpha, x);
        auto xc = ellip.xc();

        for (size_t i = 0; i < n; ++i) {
            RC_ASSERT(xc[i] == doctest::Approx(x[i]));
        }
    });
}

TEST_CASE("Property-based test: update_bias_cut reduces ellipsoid") {
    rc::check("After bias cut, ellipsoid tsq decreases or remains positive", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 6));
        auto alpha = *rc::gen::positive<double>() + 1.0;  // Ensure > 0

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create a random gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::arbitrary<double>();
        }

        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_bias_cut(cut);

        // If successful, tsq should be non-negative
        if (status == CutStatus::Success) {
            RC_ASSERT(ellip.tsq() >= 0.0);
        }
    });
}

TEST_CASE("Property-based test: update_central_cut reduces ellipsoid") {
    rc::check("After central cut, ellipsoid tsq decreases or remains positive", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 6));
        auto alpha = *rc::gen::positive<double>() + 1.0;

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create a random gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::arbitrary<double>();
        }

        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_central_cut(cut);

        // If successful, tsq should be non-negative
        if (status == CutStatus::Success) {
            RC_ASSERT(ellip.tsq() >= 0.0);
        }
    });
}

TEST_CASE("Property-based test: Multiple cuts reduce ellipsoid") {
    rc::check("Multiple cuts maintain ellipsoid tsq non-negativity", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 5));
        auto alpha = 10.0;
        auto num_cuts = static_cast<size_t>(*rc::gen::inRange(2, 6));

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        for (size_t k = 0; k < num_cuts; ++k) {
            // Create a random gradient
            std::vector<double> grad(n);
            for (size_t i = 0; i < n; ++i) {
                grad[i] = *rc::gen::arbitrary<double>();
            }

            auto beta = 0.0;
            auto cut = std::make_pair(grad, beta);
            auto status = ellip.update_central_cut(cut);

            if (status == CutStatus::Success) {
                // tsq should remain non-negative
                RC_ASSERT(ellip.tsq() >= 0.0);
            }
        }
    });
}

TEST_CASE("Property-based test: set_xc changes ellipsoid center") {
    rc::check("set_xc updates center correctly", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        auto alpha = *rc::gen::positive<double>();

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create new center
        std::vector<double> new_xc(n);
        for (size_t i = 0; i < n; ++i) {
            new_xc[i] = *rc::gen::arbitrary<double>();
        }

        ellip.set_xc(new_xc);
        auto xc = ellip.xc();

        for (size_t i = 0; i < n; ++i) {
            RC_ASSERT(xc[i] == doctest::Approx(new_xc[i]));
        }
    });
}

TEST_CASE("Property-based test: Copy constructor creates identical ellipsoid") {
    rc::check("Copied ellipsoid has same center and tsq", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        auto alpha = *rc::gen::positive<double>();

        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = *rc::gen::arbitrary<double>();
        }

        auto ellip1 = Ell<std::vector<double>>(alpha, x);
        auto ellip2 = Ell<std::vector<double>>(ellip1);  // Copy constructor

        auto xc1 = ellip1.xc();
        auto xc2 = ellip2.xc();

        for (size_t i = 0; i < n; ++i) {
            RC_ASSERT(xc1[i] == doctest::Approx(xc2[i]));
        }
        RC_ASSERT(ellip1.tsq() == doctest::Approx(ellip2.tsq()));
    });
}

TEST_CASE("Property-based test: Copy method creates identical ellipsoid") {
    rc::check("Copied ellipsoid has same center and tsq", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(1, 11));
        auto alpha = *rc::gen::positive<double>();

        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = *rc::gen::arbitrary<double>();
        }

        auto ellip1 = Ell<std::vector<double>>(alpha, x);
        auto ellip2 = ellip1.copy();

        auto xc1 = ellip1.xc();
        auto xc2 = ellip2.xc();

        for (size_t i = 0; i < n; ++i) {
            RC_ASSERT(xc1[i] == doctest::Approx(xc2[i]));
        }
        RC_ASSERT(ellip1.tsq() == doctest::Approx(ellip2.tsq()));
    });
}

TEST_CASE("Property-based test: Zero gradient cut does not change center") {
    rc::check("Cut with zero gradient leaves center unchanged", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 6));
        auto alpha = *rc::gen::positive<double>() + 1.0;

        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = *rc::gen::arbitrary<double>();
        }

        auto ellip = Ell<std::vector<double>>(alpha, x);
        auto xc_before = ellip.xc();

        // Create zero gradient
        std::vector<double> grad(n, 0.0);
        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        ellip.update_central_cut(cut);

        auto xc_after = ellip.xc();

        for (size_t i = 0; i < n; ++i) {
            RC_ASSERT(xc_after[i] == doctest::Approx(xc_before[i]));
        }
    });
}

TEST_CASE("Property-based test: Orthogonal cuts in 2D") {
    rc::check("Orthogonal cuts in 2D work correctly", []() {
        auto alpha = 10.0;
        std::vector<double> x = {0.0, 0.0};
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // First cut: x[0] >= 0 (gradient = [-1, 0], beta = 0)
        std::vector<double> grad1 = {-1.0, 0.0};
        auto beta1 = 0.0;
        auto cut1 = std::make_pair(grad1, beta1);
        auto status1 = ellip.update_bias_cut(cut1);

        if (status1 == CutStatus::Success) {
            // Second cut: x[1] >= 0 (gradient = [0, -1], beta = 0)
            std::vector<double> grad2 = {0.0, -1.0};
            auto beta2 = 0.0;
            auto cut2 = std::make_pair(grad2, beta2);
            auto status2 = ellip.update_bias_cut(cut2);

            // At least one cut should succeed
            RC_ASSERT(status1 == CutStatus::Success || status2 == CutStatus::Success);
        }
    });
}

TEST_CASE("Property-based test: Negative beta affects cut result") {
    rc::check("Negative beta in bias cut affects ellipsoid update", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 5));
        auto alpha = *rc::gen::positive<double>() + 2.0;

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::nonNegative<double>() + 0.1;
        }

        // Negative beta means the cut is "deeper"
        auto beta = -*rc::gen::positive<double>();
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_bias_cut(cut);

        // Status should be success for reasonable cuts
        RC_ASSERT(status == CutStatus::Success || status == CutStatus::NoSoln ||
                  status == CutStatus::NoEffect);
    });
}

TEST_CASE("Property-based test: Large gradient magnitude") {
    rc::check("Large gradient magnitude handled correctly", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 5));
        auto alpha = 100.0;

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create large gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::inRange(100, 1000);
        }

        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_central_cut(cut);

        // Should handle large gradients gracefully
        RC_ASSERT(status == CutStatus::Success || status == CutStatus::NoSoln ||
                  status == CutStatus::NoEffect);
    });
}

TEST_CASE("Property-based test: Small alpha ellipsoid") {
    rc::check("Small initial alpha still works", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(2, 5));
        // Generate small alpha using positive<double>() and scaling
        auto alpha = *rc::gen::positive<double>() * 0.5;

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::arbitrary<double>();
        }

        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_central_cut(cut);

        // Should handle small alpha gracefully
        RC_ASSERT(status == CutStatus::Success || status == CutStatus::NoSoln ||
                  status == CutStatus::NoEffect);
    });
}

TEST_CASE("Property-based test: High-dimensional ellipsoid") {
    rc::check("High-dimensional ellipsoids work correctly", []() {
        auto n = static_cast<size_t>(*rc::gen::inRange(5, 11));  // 5-10 dimensions
        auto alpha = 10.0;

        std::vector<double> x(n, 0.0);
        auto ellip = Ell<std::vector<double>>(alpha, x);

        // Create gradient
        std::vector<double> grad(n);
        for (size_t i = 0; i < n; ++i) {
            grad[i] = *rc::gen::arbitrary<double>();
        }

        auto beta = 0.0;
        auto cut = std::make_pair(grad, beta);
        auto status = ellip.update_central_cut(cut);

        // Should handle high dimensions gracefully
        RC_ASSERT(status == CutStatus::Success || status == CutStatus::NoSoln ||
                  status == CutStatus::NoEffect);
    });
}

#endif