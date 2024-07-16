// test_conjugate_gradient.cpp

#include <doctest/doctest.h>

#include <ellalgo/conjugate_gradient.hpp>
#include <random>

TEST_CASE("Conjugate Gradient Simple") {
    Matrix A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector b({1, 2});
    Vector x_expected({0.0909091, 0.6363636});

    Vector x = conjugate_gradient(A, b);

    CHECK((x - x_expected).norm() < 1e-5);
}

TEST_CASE("Conjugate Gradient Larger") {
    int n = 100;
    Matrix A(n, n);
    for (int i = 0; i < n; ++i) {
        A[i][i] = i + 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    Vector x_true(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = dis(gen);
    }
    Vector b = A.dot(x_true);

    Vector x = conjugate_gradient(A, b);

    CHECK((x - x_true).norm() < 1e-5);
}

TEST_CASE("Conjugate Gradient with Initial Guess") {
    Matrix A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector b({1, 2});
    Vector x0({1, 1});
    Vector x_expected({0.0909091, 0.6363636});

    Vector x = conjugate_gradient(A, b, &x0);

    CHECK((x - x_expected).norm() < 1e-5);
}

// TEST_CASE("Conjugate Gradient Non-Convergence") {
//     Matrix A(2, 2);
//     A[0] = {1, 2};
//     A[1] = {2, 1};  // Not positive definite
//     Vector b({1, 1});
//
//     CHECK_THROWS_AS(conjugate_gradient(A, b, nullptr, 1e-5, 10), std::runtime_error);
// }

TEST_CASE("Conjugate Gradient Tolerance") {
    Matrix A(2, 2);
    A[0] = {4, 1};
    A[1] = {1, 3};
    Vector b({1, 2});
    double tol = 1e-10;
    Vector x0(2);

    Vector x = conjugate_gradient(A, b, &x0, tol);

    Vector residual = b - A.dot(x);
    CHECK(residual.norm() < tol);
}
