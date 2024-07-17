#include <doctest/doctest.h>
#include <ellalgo/greeter.h>
// #include <ellalgo/version.h>

#include <string>

TEST_CASE("EllAlgo") {
    using namespace ellalgo;

    EllAlgo ellalgo("Tests");

    CHECK(ellalgo.greet(LanguageCode::EN) == "Hello, Tests!");
    CHECK(ellalgo.greet(LanguageCode::DE) == "Hallo Tests!");
    CHECK(ellalgo.greet(LanguageCode::ES) == "¡Hola Tests!");
    CHECK(ellalgo.greet(LanguageCode::FR) == "Bonjour Tests!");
}

// TEST_CASE("EllAlgo version") {
//     static_assert(std::string_view(ELLALGO_VERSION) == std::string_view("1.0"));
//     CHECK(std::string(ELLALGO_VERSION) == std::string("1.0"));
// }
