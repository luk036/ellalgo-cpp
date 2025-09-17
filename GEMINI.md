# Gemini Code Understanding

## Project Overview

This is a C++ project that implements the Ellipsoid Algorithm for linear programming. The project is built using CMake and can also be built using xmake. It is well-structured, with a clear separation between the library code, tests, and benchmarks.

The project uses `CPM.cmake` for dependency management, and the main dependencies are `fmt` and `doctest`.

## Building and Running

There are two ways to build and run the project:

### Using CMake

**Build all targets:**

```bash
cmake -S all -B build
cmake --build build
```

**Run tests:**

```bash
./build/test/EllAlgoTests
```

**Run benchmarks:**

```bash
./build/bench/BM_ell
./build/bench/BM_lmi
./build/bench/BM_lowpass
```

### Using xmake

**Build and run tests:**

```bash
xmake
xmake run test_ellalgo
```

## Development Conventions

*   **Coding Style:** The project uses `clang-format` to enforce a consistent coding style.
*   **Testing:** The project uses `doctest` for testing. Tests are located in the `test` directory.
*   **Continuous Integration:** The project uses GitHub Actions for continuous integration. The CI configuration is located in the `.github/workflows` directory.
