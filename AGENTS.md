# AGENTS.md - Development Guide for ellalgo-cpp

This file provides context for agentic coding agents working in this repository.

## Project Overview

**ellalgo-cpp** is an implementation of the Ellipsoid Method for linear/convex optimization in modern C++.
It supports parallel cuts, discrete optimization, and traditional/stable versions.

## Build Commands

### Quick Build (Test)
```bash
cmake -S test -B build/test
cmake --build build/test
```

### Run Tests
```bash
# Via CMake test target (recommended)
CTEST_OUTPUT_ON_FAILURE=1 cmake --build build/test --target test

# Or run the executable directly
./build/test/EllAlgoTests
```

### Run Single Test
```bash
# Using ctest with filter
ctest -R test_ell -V

# Or run the executable with specific test
./build/test/EllAlgoTests -tc="test_ell*"
```

### Build Everything (All targets)
```bash
cmake -S all -B build
cmake --build build

# Run tests and standalone
./build/test/EllAlgoTests
./build/standalone/EllAlgo --help
```

### Code Formatting
```bash
cmake -S test -B build/test
cmake --build build/test --target format      # check
cmake --build build/test --target fix-format # apply
```

### Build Documentation
```bash
cmake -S documentation -B build/doc
cmake --build build/doc --target GenerateDocs
```

### Additional Build Options
- Code coverage: `-DENABLE_TEST_COVERAGE=1`
- Sanitizers: `-DUSE_SANITIZER=Address`
- Static analyzers: `-DUSE_STATIC_ANALYZER=clang-tidy`

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
- **Always use**: `CXX_STANDARD 14` (or 17 for tests)

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
- **MSVC**: `/W4 /WX /wd4459 /wd4819 /wd4996`

### Dependencies (via CPM.cmake)
- `fmt` (12.1.0) - formatting
- `doctest` (2.4.11) - testing
- `rapidcheck` - property-based testing
- `PackageProject.cmake` - installation

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
├── cmake/                # CMake utilities
└── .clang-format        # Code formatting rules
```

## Important Notes

1. **No in-source builds**: Always build in separate `build/` directory
2. **Header-only warning**: The project uses mixed header/implementation pattern
3. **CPM.cmake**: Dependencies downloaded at configure time; set `CPM_SOURCE_CACHE` for offline builds
4. **Branch prediction**: Use `ELL_LIKELY` / `ELL_UNLIKELY` macros for performance-critical branches
