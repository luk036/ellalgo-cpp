#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/oracles/ldlt_mgr.hpp>  // for LDLTMgr
// #include <xtensor/xarray.hpp>          // for xarray_container
// #include <xtensor/xcontainer.hpp>      // for xcontainer<>::inner_shape_type
// #include <xtensor/xlayout.hpp>         // for layout_type,
// layout_type::row... #include <xtensor/xtensor_forward.hpp> // for xarray
// #include <xtensor/xarray.hpp>
#include <ellalgo/ell_matrix.hpp>

// using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Vec = std::valarray<double>;

TEST_CASE("Cholesky test 1") {
    Matrix m1(3U);
    m1.row(0) = Vec{25.0, 15.0, -5.0};
    m1.row(1) = Vec{15.0, 18.0, 0.0};
    m1.row(2) = Vec{-5.0, 0.0, 11.0};
    auto Q1 = LDLTMgr(3);
    CHECK(Q1.factorize(m1));
}

TEST_CASE("Cholesky test 2") {
    Matrix m2(4U);
    m2.row(0) = Vec{18.0, 22.0, 54.0, 42.0};
    m2.row(1) = Vec{22.0, -70.0, 86.0, 62.0};
    m2.row(2) = Vec{54.0, 86.0, -174.0, 134.0};
    m2.row(3) = Vec{42.0, 62.0, 134.0, -106.0};

    auto Q2 = LDLTMgr(4);
    Q2.factorize(m2);
    CHECK(!Q2.is_spd());
    // CHECK(Q2.p.second == 2);
}

TEST_CASE("Cholesky test 3") {
    Matrix m3(3U);
    m3.row(0) = Vec{0.0, 15.0, -5.0};
    m3.row(1) = Vec{15.0, 18.0, 0.0};
    m3.row(2) = Vec{-5.0, 0.0, 11.0};

    auto Q3 = LDLTMgr(3);
    Q3.factorize(m3);
    CHECK(!Q3.is_spd());
    const auto ep3 = Q3.witness();
    Vec v(0.0, 3);
    Q3.set_witness_vec(v);

    // CHECK(Q3.p.second == 1);
    CHECK_EQ(ep3, 0.0);
    CHECK_EQ(v[0], 1.0);
}
