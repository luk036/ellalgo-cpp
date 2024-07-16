// test_conjugate_gradient.cpp

#include <doctest/doctest.h>

#include <ellalgo/conjugate_gradient2.hpp>
#include <ellalgo/linear_algebra.hpp>
#include <random>

TEST_CASE("Conjugate Gradient Simple") {
    Matrix2<double> A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector2<double> b({1, 2});
    Vector2<double> x_expected({0.0909091, 0.6363636});
    Vector2<double> x0({0, 0});

    Vector2<double> x = conjugate_gradient2(A, b, &x0);

    CHECK((x - x_expected).norm() < 1e-5);
}

TEST_CASE("Conjugate Gradient Larger") {
    int n = 100;
    Matrix2<double> A(n, n);
    for (int i = 0; i < n; ++i) {
        A[i][i] = i + 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    Vector2<double> x_true(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = dis(gen);
    }
    Vector2<double> b = A * x_true;
    Vector2<double> x0(n);

    Vector2<double> x = conjugate_gradient2(A, b, &x0);

    CHECK((x - x_true).norm() < 1e-5);
}

TEST_CASE("Conjugate Gradient with Initial Guess") {
    Matrix2<double> A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector2<double> b({1, 2});
    Vector2<double> x0({1, 1});
    Vector2<double> x_expected({0.0909091, 0.6363636});

    Vector2<double> x = conjugate_gradient2(A, b, &x0);

    CHECK((x - x_expected).norm() < 1e-5);
}

// TEST_CASE("Conjugate Gradient Non-Convergence") {
//     Matrix2<double> A(2, 2);
//     A[0] = {1, 2};
//     A[1] = {2, 1};  // Not positive definite
//     Vector2<double> b({1, 1});
//     Vector2<double> x0(2);
//
//     CHECK_THROWS_AS(conjugate_gradient2(A, b, &x0, 1e-5, 10), std::runtime_error);
// }

TEST_CASE("Conjugate Gradient Tolerance") {
    Matrix2<double> A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector2<double> b({1, 2});
    double tol = 1e-10;
    Vector2<double> x0(2);

    Vector2<double> x = conjugate_gradient2(A, b, &x0, tol);

    Vector2<double> residual = b - A * x;
    CHECK(residual.norm() < tol);
}
