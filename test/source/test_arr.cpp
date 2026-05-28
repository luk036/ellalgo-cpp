#include <doctest/doctest.h>

#include <algorithm>  // for std::sort
#include <cmath>
#include <ellalgo/arr.hpp>

TEST_CASE("Arr: 1D construction") {
    Arr a(5);
    CHECK(a.size() == 5);
    CHECK(a.rows() == 5);
    CHECK(!a.is_2d());

    Arr b(3);
    b(0) = 1.5;
    b(1) = 1.5;
    b(2) = 1.5;
    CHECK(b(0) == 1.5);
    CHECK(b(2) == 1.5);
}

TEST_CASE("Arr: 2D construction") {
    Arr a(3, 4);
    CHECK(a.rows() == 3);
    CHECK(a.cols() == 4);
    CHECK(a.size() == 12);
    CHECK(a.is_2d());

    a(1, 2) = 3.14;
    CHECK(a(1, 2) == 3.14);
}

TEST_CASE("Arr: initializer list") {
    auto a = Arr{1.0, 2.0, 3.0};
    CHECK(a.size() == 3);
    CHECK(a(0) == 1.0);
    CHECK(a(2) == 3.0);
}

TEST_CASE("Arr: zeros and ones") {
    auto z = zeros(4);
    CHECK(z.size() == 4);
    for (size_t i = 0; i < 4; ++i) CHECK(z(i) == 0.0);

    auto z2d = zeros(2, 3);
    CHECK(z2d.rows() == 2);
    CHECK(z2d.cols() == 3);

    auto o = ones(2, 2);
    CHECK(o(0, 0) == 1.0);
    CHECK(o(1, 1) == 1.0);
}

TEST_CASE("Arr: linspace") {
    auto w = linspace(0.0, 1.0, 5);
    CHECK(w.size() == 5);
    CHECK(w(0) == 0.0);
    CHECK(w(4) == 1.0);
    CHECK(w(2) == 0.5);
}

TEST_CASE("Arr: arange") {
    auto a = arange(0.0, 4.0);
    CHECK(a.size() == 4);
    CHECK(a(0) == 0.0);
    CHECK(a(3) == 3.0);
}

TEST_CASE("Arr: element-wise arithmetic + - *") {
    auto a = Arr{1.0, 2.0, 3.0};
    auto b = Arr{4.0, 5.0, 6.0};

    auto s = a + b;
    CHECK(s(0) == 5.0);
    CHECK(s(2) == 9.0);

    auto d = b - a;
    CHECK(d(0) == 3.0);
    CHECK(d(1) == 3.0);

    auto p = a * b;
    CHECK(p(0) == 4.0);
    CHECK(p(2) == 18.0);
}

TEST_CASE("Arr: scalar arithmetic") {
    auto a = Arr{1.0, 2.0, 3.0};
    auto m = 2.0 * a;
    CHECK(m(0) == 2.0);
    CHECK(m(2) == 6.0);

    auto n = a / 2.0;
    CHECK(n(0) == 0.5);
}

TEST_CASE("Arr: unary minus") {
    auto a = Arr{1.0, -2.0, 3.0};
    auto n = -a;
    CHECK(n(0) == -1.0);
    CHECK(n(1) == 2.0);
}

TEST_CASE("Arr: in-place += and -=") {
    auto a = Arr{1.0, 2.0};
    a += 1.0;
    CHECK(a(0) == 2.0);
    a -= 0.5;
    CHECK(a(1) == 2.5);
}

TEST_CASE("Arr: make_same_shape") {
    auto a1d = Arr(5);
    auto s1 = make_same_shape(a1d);
    CHECK(s1.size() == 5);
    CHECK(!s1.is_2d());

    auto a2d = Arr(3, 4);
    auto s2 = make_same_shape(a2d);
    CHECK(s2.rows() == 3);
    CHECK(s2.cols() == 4);
    CHECK(s2.is_2d());
}

TEST_CASE("Arr: element-wise cos/log/abs/exp") {
    auto a = linspace(0.0, 1.0, 3);
    auto c = cos(a);
    CHECK(c(0) == doctest::Approx(std::cos(0.0)));
    CHECK(c(2) == doctest::Approx(std::cos(1.0)));

    auto e = exp(Arr{0.0, 1.0});
    CHECK(e(0) == 1.0);
    CHECK(e(1) == doctest::Approx(std::exp(1.0)));

    auto n = abs(Arr{-2.0, 3.0});
    CHECK(n(0) == 2.0);
    CHECK(n(1) == 3.0);
}

TEST_CASE("Arr: sum") {
    auto a = Arr{1.0, 2.0, 3.0};
    CHECK(sum(a) == 6.0);
}

TEST_CASE("Arr: comparison operators") {
    auto a = linspace(0.0, 5.0, 6);

    auto le = a <= 2.0;
    CHECK(le(0) == 1.0);
    CHECK(le(2) == 1.0);
    CHECK(le(3) == 0.0);

    auto ge = a >= 4.0;
    CHECK(ge(4) == 1.0);
    CHECK(ge(2) == 0.0);

    auto lt = a < 1.0;
    CHECK(lt(0) == 1.0);
    CHECK(lt(1) == 0.0);

    auto gt = a > 3.0;
    CHECK(gt(3) == 0.0);
    CHECK(gt(4) == 1.0);
}

TEST_CASE("Arr: where") {
    auto a = Arr{0.0, 1.0, 0.0, 2.0, 0.0};
    auto idx = where(a > 0.5)[0];
    CHECK(idx.size() == 2);
    CHECK(idx(0) == 1.0);
    CHECK(idx(1) == 3.0);
}

TEST_CASE("Arr: dot product (matrix-vector)") {
    // A = [[1,2],[3,4]], x = [5,6]
    auto A = Arr(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    auto x = Arr{5.0, 6.0};

    auto y = dot(A, x);
    CHECK(y.size() == 2);
    CHECK(y(0) == 1 * 5 + 2 * 6);
    CHECK(y(1) == 3 * 5 + 4 * 6);
}

TEST_CASE("Arr: outer product") {
    auto u = Arr{1.0, 2.0};
    auto v = Arr{3.0, 4.0, 5.0};
    auto o = outer(u, v);

    CHECK(o.rows() == 2);
    CHECK(o.cols() == 3);
    CHECK(o(0, 0) == 3.0);
    CHECK(o(0, 1) == 4.0);
    CHECK(o(1, 2) == 10.0);
}

TEST_CASE("Arr: concatenate") {
    auto a = ones(2, 2);
    auto b = zeros(2, 1);
    auto c = concatenate(a, b, 1);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 3);
    CHECK(c(0, 0) == 1.0);
    CHECK(c(0, 2) == 0.0);
}

TEST_CASE("Arr: view") {
    auto A = Arr(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) A(i, j) = double(i * 4 + j);

    auto sub = view(A, Range(1, 3), Range(0, 2));
    CHECK(sub.rows() == 2);
    CHECK(sub.cols() == 2);
    CHECK(sub(0, 0) == 4.0);
    CHECK(sub(1, 1) == 9.0);
}

TEST_CASE("Arr: view with ALL") {
    auto A = ones(3, 2);
    auto col0 = view(A, Range(Range::ALL), Range(0, 1));
    CHECK(col0.rows() == 3);
    CHECK(col0.cols() == 1);
}

TEST_CASE("Arr: strided view") {
    auto a = linspace(0.0, 10.0, 11);
    auto step2 = view(a, Range(0, 11, 2));
    CHECK(step2.size() == 6);
    CHECK(step2(0) == 0.0);
    CHECK(step2(1) == 2.0);
    CHECK(step2(5) == 10.0);
}

TEST_CASE("Arr: eval") {
    auto a = Arr{1.0, 2.0};
    auto b = eval(a);
    CHECK(b(0) == 1.0);
    b(0) = 99.0;
    CHECK(a(0) == 1.0);
}

TEST_CASE("Arr: from vector") {
    std::vector<double> v = {1.0, 2.0, 3.0};
    Arr a(v);
    CHECK(a.size() == 3);
    CHECK(a(1) == 2.0);
}

TEST_CASE("Arr: operator[] for EllAlgo compatibility") {
    Arr a(4);
    a[0] = 10.0;
    a[3] = 40.0;
    CHECK(a[0] == 10.0);
    CHECK(a[3] == 40.0);
}

TEST_CASE("Arr: data pointer") {
    Arr a(3);
    a(0) = 1.0;
    a(1) = 1.0;
    a(2) = 1.0;
    auto* ptr = a.data();
    CHECK(ptr[0] == 1.0);
    ptr[1] = 2.0;
    CHECK(a(1) == 2.0);
}

TEST_CASE("Arr: begin/end") {
    Arr a = Arr{3.0, 1.0, 2.0};
    std::sort(a.begin(), a.end());
    CHECK(a(0) == 1.0);
    CHECK(a(2) == 3.0);
}
