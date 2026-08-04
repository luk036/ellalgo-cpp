# AGENTS.md - Development Guide for ellalgo-cpp

This file provides context for agentic coding agents working in this repository.

## Project Overview

**ellalgo-cpp** is an implementation of the Ellipsoid Method for linear/convex optimization in modern C++.
It supports parallel cuts, discrete optimization, and traditional/stable versions.

## Build Commands

### Quick Build
```bash
cmake -B build
cmake --build build
```

### Run Tests
```bash
# Via ctest (recommended)
ctest --test-dir build --output-on-failure

# Or run the executable directly
./build/EllAlgoTests
```

### Run Single Test
```bash
# Using ctest with filter
ctest --test-dir build -R test_ell -V

# Or run the executable with specific test
./build/EllAlgoTests -tc="test_ell*"
```

### Build Everything
```bash
cmake -B build
cmake --build build

# Run tests and standalone
./build/EllAlgoTests
./build/EllAlgo --help
```

### Code Formatting
```bash
# Requires clang-format
clang-format -i include/ source/ test/source/ standalone/source/ bench/*.cpp
```

### Build Documentation
```bash
cmake -B build -DELLALGO_BUILD_DOCS=ON
cmake --build build --target GenerateDocs
```

### Additional Build Options
- Code coverage: `-DELLALGO_ENABLE_COVERAGE=1` (adds a `coverage` gcovr target on GCC/Clang)
- clang-tidy: `-DELLALGO_ENABLE_CLANG_TIDY=ON` then `cmake --build build --target clang-tidy`
- Benchmarks: `-DELLALGO_BUILD_BENCHMARKS=ON`

## Code Style Guidelines

### Formatting
- **Style**: Google with modifications (see `.clang-format`)
- **Column limit**: 100
- **Indent width**: 4 spaces
- **Brace style**: Attach
- **Namespace indentation**: All

### C++ Standards
- **Library**: C++14
- **Tests**: C++17
- **Always use**: `CXX_STANDARD 20` (or 17 for tests)

### Naming Conventions
- **Classes**: PascalCase (`Ell`, `EllCalc`, `CutStatus`)
- **Functions**: snake_case or camelCase depending on context
- **Member variables**: Leading underscore + snake_case (`_xc`, `_mgr`, `_n`)
- **Constants**: kCamelCase or SCREAMING_SNAKE_CASE
- **Files**: lowercase with underscores (`ell.hpp`, `ell_calc.cpp`)

### Code Patterns

#### Include Order (via clang-format IncludeBlocks: Regroup):
1. Standard library (`<cmath>`, `<vector>`, etc.)
2. Related header (`.hpp`/`.h`)
3. Other project headers (`<ellalgo/...>`)
4. External dependencies

#### Function Return Types
```cpp
// Use trailing return type for class methods
auto calc_bias_cut(const double beta, const double tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>>;
```

#### Member Access
```cpp
// Use this-> for member access
auto xc() const -> Arr { return this->_xc; }
this->_mgr.update_bias_cut(grad, beta);
```

#### Error Handling
```cpp
// Use CutStatus enum for algorithm status
enum class CutStatus { Success, NoSoln, NoEffect, Infinity };

if (ELL_UNLIKELY(eta <= 0.0)) {
    return {CutStatus::NoEffect, {0.0, 0.0, 1.0}};
}
```

### Documentation
- Use Doxygen-style comments for classes and functions
- Document parameters with `@param[in]` / `@param[out]`
- Use `@return` for return value documentation

### Testing
- **Framework**: doctest
- **Test file naming**: `test_*.cpp`
- **Test cases**: `TEST_CASE("Description")` / `SUBCASE`
- **Assertions**: `CHECK_EQ`, `CHECK_NE`, `REQUIRE`

### Compiler Flags (Enforced)
- **GCC/Clang**: `-Wall -Wpedantic -Wextra -Werror`
- **MSVC**: `/utf-8 /W4 /WX`

### Dependencies (via FetchContent or system packages)
- `doctest` (2.4.11) - testing (system-installed first, header download fallback)
- `spdlog` (v1.17.0) - logging (bundles fmt)
- `cxxopts` (v3.2.1) - CLI parsing for the standalone example
- `nanobench` (v4.3.11) - benchmarking (only with `ELLALGO_BUILD_BENCHMARKS`)

## Project Structure

```
ellalgo-cpp/
├── include/ellalgo/       # Header files (.hpp)
│   ├── ell.hpp           # Main ellipsoid class
│   ├── ell_calc.hpp      # Calculation utilities
│   ├── ell_core.hpp      # Core implementation
│   ├── ell_assert.hpp    # Branch prediction macros
│   └── oracles/         # Oracle implementations
├── source/               # Implementation files (.cpp)
├── test/source/          # Test source files
├── standalone/source/    # Example executable
├── bench/                # Benchmarks
├── documentation/        # Doxygen config and pages
├── test_installed/       # find_package consumer test (CI only)
├── CMakeLists.txt        # Single build configuration
└── .clang-format        # Code formatting rules
```

## Important Notes

1. **No in-source builds**: Always build in separate `build/` directory
2. **Header-only warning**: The project uses mixed header/implementation pattern
3. **FetchContent**: Dependencies downloaded at configure time; set `FETCHCONTENT_SOURCE_DIR_*` or a local git cache for offline builds
4. **Branch prediction**: Use `ELL_LIKELY` / `ELL_UNLIKELY` macros for performance-critical branches
