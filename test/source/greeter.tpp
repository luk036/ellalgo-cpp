#include <doctest/doctest.h>
#include <ellalgo/greeter.h>
// #include <ellalgo/version.h>

#include <string>

TEST_CASE("EllAlgo") {
    using namespace ellalgo;

    EllAlgo ellalgo("Tests");

    CHECK_EQ(ellalgo.greet(LanguageCode::EN), "Hello, Tests!");
    CHECK_EQ(ellalgo.greet(LanguageCode::DE), "Hallo Tests!");
    CHECK_EQ(ellalgo.greet(LanguageCode::ES), "¡Hola Tests!");
    CHECK_EQ(ellalgo.greet(LanguageCode::FR), "Bonjour Tests!");
}

// TEST_CASE("EllAlgo version") {
//     static_assert(std::string_view(ELLALGO_VERSION) == std::string_view("1.0"));
//     CHECK_EQ(std::string(ELLALGO_VERSION), std::string("1.0"));
// }
