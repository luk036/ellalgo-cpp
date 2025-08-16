#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/ell_matrix.hpp>
#include <ellalgo/oracles/ldlt_mgr.hpp>  // for LDLTMgr

using Vec = std::valarray<double>;

TEST_CASE("Cholesky test 1") {
    Matrix m1(3U);
    m1.row(0) = Vec{25.0, 15.0, -5.0};
    m1.row(1) = Vec{15.0, 18.0, 0.0};
    m1.row(2) = Vec{-5.0, 0.0, 11.0};
    auto ldlt_mgr = LDLTMgr(3);
    CHECK(ldlt_mgr.factorize(m1));
}

TEST_CASE("Cholesky test 2") {
    Matrix m2(4U);
    m2.row(0) = Vec{18.0, 22.0, 54.0, 42.0};
    m2.row(1) = Vec{22.0, -70.0, 86.0, 62.0};
    m2.row(2) = Vec{54.0, 86.0, -174.0, 134.0};
    m2.row(3) = Vec{42.0, 62.0, 134.0, -106.0};

    auto ldlt_mgr = LDLTMgr(4);
    ldlt_mgr.factorize(m2);
    CHECK(!ldlt_mgr.is_spd());
    ldlt_mgr.witness();
    CHECK_EQ(ldlt_mgr.pos, std::pair<size_t, size_t>{0, 2});
}

TEST_CASE("Cholesky test 3") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, 11.0};

    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factorize(m3);
    CHECK(!ldlt_mgr.is_spd());
    const auto ep = ldlt_mgr.witness();
    Vec v(0.0, 3);
    ldlt_mgr.set_witness_vec(v);

    CHECK_EQ(ldlt_mgr.pos, std::pair<size_t, size_t>{0, 1});
    CHECK_EQ(v[0], 1.0);
    CHECK_EQ(ep, 0.0);
}

TEST_CASE("Cholesky test 4") {
    Matrix m1(3U);
    m1.row(0) = Vec{25.0, 15.0, -5.0};
    m1.row(1) = Vec{15.0, 18.0, 0.0};
    m1.row(2) = Vec{-5.0, 0.0, 11.0};
    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factor_with_allow_semidefinite([&m1](size_t i, size_t j) { return m1(i, j); });
    CHECK(ldlt_mgr.is_spd());
}

TEST_CASE("Cholesky test 5") {
    Matrix m2(4U);
    m2.row(0) = Vec{18.0, 22.0, 54.0, 42.0};
    m2.row(1) = Vec{22.0, -70.0, 86.0, 62.0};
    m2.row(2) = Vec{54.0, 86.0, -174.0, 134.0};
    m2.row(3) = Vec{42.0, 62.0, 134.0, -106.0};

    auto ldlt_mgr = LDLTMgr(4);
    ldlt_mgr.factor_with_allow_semidefinite([&m2](size_t i, size_t j) { return m2(i, j); });
    CHECK(!ldlt_mgr.is_spd());
    ldlt_mgr.witness();
    CHECK_EQ(ldlt_mgr.pos, std::pair<size_t, size_t>{0, 2});
}

TEST_CASE("Cholesky test 6") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, 11.0};

    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factor_with_allow_semidefinite([&m3](size_t i, size_t j) { return m3(i, j); });
    CHECK(ldlt_mgr.is_spd());
}

TEST_CASE("Cholesky test 7") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, -20.0};

    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factor_with_allow_semidefinite([&m3](size_t i, size_t j) { return m3(i, j); });
    CHECK(!ldlt_mgr.is_spd());
    const auto ep = ldlt_mgr.witness();
    CHECK_EQ(ep, 20.0);
}

TEST_CASE("Cholesky test 8") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, 20.0};

    auto ldlt_mgr = LDLTMgr(3);
    CHECK(!ldlt_mgr.factorize(m3));
}

TEST_CASE("Cholesky test 9") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, 20.0};

    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factor_with_allow_semidefinite([&m3](size_t i, size_t j) { return m3(i, j); });
    CHECK(ldlt_mgr.is_spd());
}

TEST_CASE("Cholesky test sqrt") {
    Matrix mat(3U);
    mat.row(0) = Vec{1.0, 0.5, 0.5};
    mat.row(1) = Vec{0.5, 1.25, 0.75};
    mat.row(2) = Vec{0.5, 0.75, 1.5};

    auto ldlt_mgr = LDLTMgr(3);
    ldlt_mgr.factor([&mat](size_t i, size_t j) { return mat(i, j); });
    CHECK(ldlt_mgr.is_spd());

    Matrix R(3);
    ldlt_mgr.sqrt(R);
    CHECK_EQ(R(0, 0), doctest::Approx(1.0));
    CHECK_EQ(R(0, 1), doctest::Approx(0.5));
    CHECK_EQ(R(0, 2), doctest::Approx(0.5));
    CHECK_EQ(R(1, 0), doctest::Approx(0.0));
    CHECK_EQ(R(1, 1), doctest::Approx(1.0));
    CHECK_EQ(R(1, 2), doctest::Approx(0.5));
    CHECK_EQ(R(2, 0), doctest::Approx(0.0));
    CHECK_EQ(R(2, 1), doctest::Approx(0.0));
    CHECK_EQ(R(2, 2), doctest::Approx(1.0));
}

TEST_CASE("Cholesky test 10") {
    Matrix m(5U);
    m.row(0) = Vec{4.0, 1.0, 1.0, 1.0, 1.0};
    m.row(1) = Vec{1.0, 5.0, 1.0, 1.0, 1.0};
    m.row(2) = Vec{1.0, 1.0, 6.0, 1.0, 1.0};
    m.row(3) = Vec{1.0, 1.0, 1.0, 7.0, 1.0};
    m.row(4) = Vec{1.0, 1.0, 1.0, 1.0, 8.0};

    auto ldlt_mgr = LDLTMgr(5);
    CHECK(ldlt_mgr.factorize(m));
    CHECK(ldlt_mgr.is_spd());
}

// TEST_CASE("Cholesky test solve") {
//     Matrix m(3U);
//     m.row(0) = Vec{25.0, 15.0, -5.0};
//     m.row(1) = Vec{15.0, 18.0, 0.0};
//     m.row(2) = Vec{-5.0, 0.0, 11.0};
//
//     auto ldlt_mgr = LDLTMgr(3);
//     CHECK(ldlt_mgr.factorize(m));
//
//     Vec b{1.0, 2.0, 3.0};
//     Vec x(0.0, 3);
//     ldlt_mgr.solve(b, x);
//
//     CHECK_EQ(x[0], doctest::Approx(0.07692307692307693));
//     CHECK_EQ(x[1], doctest::Approx(0.038461538461538464));
//     CHECK_EQ(x[2], doctest::Approx(0.3076923076923077));
// }
