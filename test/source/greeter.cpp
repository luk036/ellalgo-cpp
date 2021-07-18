#include <doctest/doctest.h>
#include <ell/greeter.h>
#include <ell/version.h>

#include <string>

TEST_CASE("Ell") {
  using namespace ell;

  Ell ell("Tests");

  CHECK(ell.greet(LanguageCode::EN) == "Hello, Tests!");
  CHECK(ell.greet(LanguageCode::DE) == "Hallo Tests!");
  CHECK(ell.greet(LanguageCode::ES) == "¡Hola Tests!");
  CHECK(ell.greet(LanguageCode::FR) == "Bonjour Tests!");
}

TEST_CASE("Ell version") {
  static_assert(std::string_view(ELL_VERSION) == std::string_view("1.0"));
  CHECK(std::string(ELL_VERSION) == std::string("1.0"));
}
