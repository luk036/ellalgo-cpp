#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/ell_matrix.hpp>

using Vec = std::valarray<double>;

TEST_CASE("Ell Matrix test 1") {
    Matrix m1(3U);
    m1.row(0) = Vec{25.0, 15.0, -5.0};
    m1.row(1) = Vec{15.0, 18.0, 0.0};
    m1.row(2) = Vec{-5.0, 0.0, 11.0};
    CHECK_EQ(m1(0, 1), 15.0);
    CHECK_EQ(m1(1, 2), 0.0);
    CHECK_EQ(m1(2, 0), -5.0);

    m1.diagonal() = Vec{4.0, 5.0, 6.0};
    CHECK_EQ(m1(0, 0), 4.0);
    CHECK_EQ(m1(1, 1), 5.0);
    CHECK_EQ(m1(2, 2), 6.0);

    CHECK_EQ(Vec(m1.row(1))[0], 15.0);
    CHECK_EQ(Vec(m1.row(1))[1], 5.0);
    CHECK_EQ(Vec(m1.row(1))[2], 0.0);
    CHECK_EQ(Vec(m1.row(2))[0], -5.0);
    CHECK_EQ(Vec(m1.row(2))[1], 0.0);
    CHECK_EQ(Vec(m1.row(2))[2], 6.0);
    CHECK_EQ(Vec(m1.diagonal())[0], 4.0);
    CHECK_EQ(Vec(m1.diagonal())[1], 5.0);
    CHECK_EQ(Vec(m1.diagonal())[2], 6.0);
}

TEST_CASE("Ell Matrix test 2") {
    Matrix m2(4U);
    m2.row(0) = Vec{18.0, 22.0, 54.0, 42.0};
    m2.row(1) = Vec{22.0, -70.0, 86.0, 62.0};
    m2.row(2) = Vec{54.0, 86.0, -174.0, 134.0};
    m2.row(3) = Vec{42.0, 62.0, 134.0, -106.0};
    CHECK_EQ(Vec(m2.row(0))[0], 18.0);
    CHECK_EQ(Vec(m2.row(0))[1], 22.0);
    CHECK_EQ(Vec(m2.row(0))[2], 54.0);
    CHECK_EQ(Vec(m2.row(0))[3], 42.0);
    CHECK_EQ(Vec(m2.row(1))[0], 22.0);
    CHECK_EQ(Vec(m2.row(1))[1], -70.0);
    CHECK_EQ(Vec(m2.row(1))[2], 86.0);
    CHECK_EQ(Vec(m2.row(1))[3], 62.0);
    CHECK_EQ(Vec(m2.row(2))[0], 54.0);
    CHECK_EQ(Vec(m2.row(2))[1], 86.0);
    CHECK_EQ(Vec(m2.row(2))[2], -174.0);
    CHECK_EQ(Vec(m2.row(2))[3], 134.0);
    CHECK_EQ(Vec(m2.row(3))[0], 42.0);
    CHECK_EQ(Vec(m2.row(3))[1], 62.0);
    CHECK_EQ(Vec(m2.row(3))[2], 134.0);
    CHECK_EQ(Vec(m2.row(3))[3], -106.0);
    CHECK_EQ(Vec(m2.diagonal())[0], 18.0);
    CHECK_EQ(Vec(m2.diagonal())[1], -70.0);
    CHECK_EQ(Vec(m2.diagonal())[2], -174.0);
    CHECK_EQ(Vec(m2.diagonal())[3], -106.0);
}
