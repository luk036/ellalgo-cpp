# AGENTS.md - Development Guide for ellalgo-cpp

This file provides context for agentic coding agents working in this repository.

## Project Overview

**ellalgo-cpp** is an implementation of the Ellipsoid Method for linear/convex optimization in modern C++.
It supports parallel cuts, discrete optimization, and traditional/stable versions.

## Build Commands

### Quick Build (All targets)
```bash
cmake -B build
cmake --build build
```

### Run Tests
```bash
# Via CMake test target (recommended)
CTEST_OUTPUT_ON_FAILURE=1 cmake --build build --target test

# Or run the executable directly
./build/test_all
```

### Run Single Test
```bash
# Using ctest with filter
ctest -R test_ell -V

# Or run the executable with specific test
./build/test_all -tc="test_ell*"
```

### Build Benchmarks
```bash
cmake -B build -DELLALGO_BUILD_BENCHMARKS=ON
cmake --build build
```

### Build Standalone
```bash
cmake -B build -DELLALGO_BUILD_STANDALONE=ON
cmake --build build
./build/ellalgo_standalone --help
```

### Build Documentation
```bash
cmake -B build -DELLALGO_BUILD_DOCS=ON
cmake --build build --target doxygen
```

### Additional Build Options
- Code coverage: `-DELLALGO_ENABLE_COVERAGE=1` (GCC/Clang only, requires gcovr)

## Code Style Guidelines

### Formatting
- **Style**: Google with modifications (see `.clang-format`)
- **Column limit**: 100
- **Indent width**: 4 spaces
- **Brace style**: Attach
- **Namespace indentation**: All

### C++ Standards
- **Library**: C++20
- **Tests**: C++20

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
- **Framework**: doctest (primary), RapidCheck (property-based)
- **Test file naming**: `test_*.cpp`
- **Test cases**: `TEST_CASE("Description")` / `SUBCASE`
- **Assertions**: `CHECK_EQ`, `CHECK_NE`, `REQUIRE`

### Compiler Flags (Enforced)
- **GCC/Clang**: `-Wall -Wpedantic -Wextra -Werror`
- **MSVC**: `/utf-8 /W4 /WX`

### Dependencies (via FetchContent / file download)
- `spdlog` (v1.17.0) - logging
- `doctest` (2.5.2) - testing
- `nanobench` (v4.3.11) - microbenchmarking
- `cxxopts` (v3.2.1) - CLI option parsing (standalone only)

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
├── test/                # Test suite
│   └── source/           # Test source files
├── standalone/           # Example executable
├── bench/                # Benchmarks
├── documentation/        # Doxygen config
├── .clang-format        # Code formatting rules
```

## Important Notes

1. **No in-source builds**: Always build in separate `build/` directory
2. **Header-only warning**: The project uses mixed header/implementation pattern
3. **Dependencies**: Downloaded via FetchContent at configure time (spdlog, nanobench, cxxopts) or file download (doctest)
4. **Branch prediction**: Use `ELL_LIKELY` / `ELL_UNLIKELY` macros for performance-critical branches
