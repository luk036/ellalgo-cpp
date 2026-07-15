#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/ell1d.hpp>     // for ell1d
#include <ellalgo/ell_config.hpp> // for CutStatus
#include <utility>                // for pair

TEST_CASE("ell1d: construction with bounds") {
    auto e = ell1d(0.0, 10.0);
    CHECK_EQ(e.xc(), 5.0);
    CHECK_EQ(e.tsq(), 0.0);
}

TEST_CASE("ell1d: construction with negative bounds") {
    auto e = ell1d(-5.0, 5.0);
    CHECK_EQ(e.xc(), 0.0);
    CHECK_EQ(e.tsq(), 0.0);
}

TEST_CASE("ell1d: construction with zero-width interval") {
    auto e = ell1d(3.0, 3.0);
    CHECK_EQ(e.xc(), 3.0);
    CHECK_EQ(e.tsq(), 0.0);
}

TEST_CASE("ell1d: set_xc") {
    auto e = ell1d(0.0, 10.0);
    e.set_xc(7.0);
    CHECK_EQ(e.xc(), 7.0);
}

TEST_CASE("ell1d: update_central_cut with positive gradient") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    auto status = e.update_central_cut(std::make_pair(1.0, 0.0));
    CHECK_EQ(status, CutStatus::Success);
    // After: tau = 5, _r = 2.5, _xc = 5 - 2.5 = 2.5
    CHECK_EQ(e.xc(), 2.5);
    CHECK_EQ(e.tsq(), 25.0);
}

TEST_CASE("ell1d: update_central_cut with negative gradient") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    auto status = e.update_central_cut(std::make_pair(-1.0, 0.0));
    CHECK_EQ(status, CutStatus::Success);
    // After: tau = 5, _r = 2.5, _xc = 5 + 2.5 = 7.5
    CHECK_EQ(e.xc(), 7.5);
    CHECK_EQ(e.tsq(), 25.0);
}

TEST_CASE("ell1d: update (deep cut) success with positive gradient") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    // cut = (g=1, beta=2): tau = |5*1| = 5, -tau=-5 < beta=2 < tau=5 → success
    auto status = e.update(std::make_pair(1.0, 2.0));
    CHECK_EQ(status, CutStatus::Success);
    // bound = 5 - 2/1 = 3, g>0 so u=3, l=5-5=0
    // _r = half_nonnegative(3-0) = 1.5, _xc = 0 + 1.5 = 1.5
    CHECK_EQ(e.xc(), 1.5);
}

TEST_CASE("ell1d: update (deep cut) success with negative gradient") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    // cut = (g=-1, beta=2): tau = |5*-1| = 5, -tau=-5 < beta=2 < tau=5 → success
    auto status = e.update(std::make_pair(-1.0, 2.0));
    CHECK_EQ(status, CutStatus::Success);
    // bound = 5 - 2/(-1) = 7, g<0 so u=5+5=10, l=7
    // _r = half_nonnegative(10-7) = 1.5, _xc = 7 + 1.5 = 8.5
    CHECK_EQ(e.xc(), 8.5);
}

TEST_CASE("ell1d: update (deep cut) NoSoln when beta > tau") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    // cut = (g=1, beta=10): tau = |5*1| = 5, beta=10 > tau=5 → NoSoln
    auto status = e.update(std::make_pair(1.0, 10.0));
    CHECK_EQ(status, CutStatus::NoSoln);
    // State should be unchanged
    CHECK_EQ(e.xc(), 5.0);
}

TEST_CASE("ell1d: update (deep cut) NoEffect when beta < -tau") {
    auto e = ell1d(0.0, 10.0);
    // Initial: _r = 5, _xc = 5
    // cut = (g=1, beta=-10): tau = |5*1| = 5, beta=-10 < -tau=-5 → NoEffect
    auto status = e.update(std::make_pair(1.0, -10.0));
    CHECK_EQ(status, CutStatus::NoEffect);
    // State should be unchanged
    CHECK_EQ(e.xc(), 5.0);
}

TEST_CASE("ell1d: multiple updates sequence") {
    auto e = ell1d(0.0, 10.0);
    CHECK_EQ(e.xc(), 5.0);

    // First central cut (g > 0) → shift left
    e.update_central_cut(std::make_pair(1.0, 0.0));
    CHECK_EQ(e.xc(), 2.5);

    // Second central cut (g < 0) → shift right
    e.update_central_cut(std::make_pair(-1.0, 0.0));
    CHECK_EQ(e.xc(), 3.75);

    // Third central cut (g > 0) → shift left
    e.update_central_cut(std::make_pair(1.0, 0.0));
    CHECK_EQ(e.xc(), 3.125);
}

TEST_CASE("ell1d: update with near-boundary beta") {
    auto e = ell1d(0.0, 10.0);

    // beta exactly at tau boundary — should succeed (beta <= tau)
    // tau = |5*1| = 5, beta = 5
    auto status = e.update(std::make_pair(1.0, 5.0));
    CHECK_EQ(status, CutStatus::Success);

    // Reset
    e = ell1d(0.0, 10.0);

    // beta exactly at -tau boundary — should be NoEffect (beta < -tau)
    // tau = |5*1| = 5, beta = -5 → beta = -5, -tau = -5 → beta is not < -tau
    // Actually -5 < -5 is false, so this should succeed
    status = e.update(std::make_pair(1.0, -5.0));
    CHECK_EQ(status, CutStatus::Success);
}
