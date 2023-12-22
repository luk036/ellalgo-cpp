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

    // CHECK_EQ(Vec{m1.row(1)}, Vec{15.0, 18.0, 0.0});
    // CHECK_EQ(Vec{m1.row(2)}, Vec{-5.0, 0.0, 11.0});
    // CHECK_EQ(Vec{m1.diagonal()}, Vec{25.0, 18.0, 11.0});
}

TEST_CASE("Ell Matrix test 2") {
    Matrix m2(4U);
    m2.row(0) = Vec{18.0, 22.0, 54.0, 42.0};
    m2.row(1) = Vec{22.0, -70.0, 86.0, 62.0};
    m2.row(2) = Vec{54.0, 86.0, -174.0, 134.0};
    m2.row(3) = Vec{42.0, 62.0, 134.0, -106.0};
    // CHECK_EQ(Vec{m2.row(0)}, Vec{18.0, 22.0, 54.0, 42.0});
    // CHECK_EQ(Vec{m2.row(1)}, Vec{22.0, -70.0, 86.0, 62.0});
    // CHECK_EQ(Vec{m2.row(2)}, Vec{54.0, 86.0, -174.0, 134.0});
    // CHECK_EQ(Vec{m2.row(3)}, Vec{42.0, 62.0, 134.0, -106.0});
    // CHECK_EQ(Vec{m2.diagonal()}, Vec{18.0, -70.0, -174.0, -106.0});
}
