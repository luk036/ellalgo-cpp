[![MacOS Build Status](https://github.com/luk036/ellalgo-cpp/workflows/MacOS/badge.svg)](https://github.com/luk036/ellalgo-cpp/actions)
[![Windows Build Status](https://github.com/luk036/ellalgo-cpp/workflows/Windows/badge.svg)](https://github.com/luk036/ellalgo-cpp/actions)
[![Ubuntu Build Status](https://github.com/luk036/ellalgo-cpp/workflows/Ubuntu/badge.svg)](https://github.com/luk036/ellalgo-cpp/actions)
[![Install Status](https://github.com/luk036/ellalgo-cpp/workflows/Install/badge.svg)](https://github.com/luk036/ellalgo-cpp/actions)
[![codecov](https://codecov.io/gh/luk036/ellalgo-cpp/branch/master/graph/badge.svg)](https://codecov.io/gh/luk036/ellalgo-cpp)

<p align="center">
  <img src="./ellalgo.svg"/>
</p>

# 👁️ ellalgo-cpp

Ellipsoid Algorithm in Modern C++
The Ellipsoid Method as a linear programming algorithm was first introduced by L. G. Khachiyan in 1979. It is a polynomial-time algorithm that uses ellipsoids to iteratively reduce the feasible region of a linear program until an optimal solution is found. The method works by starting with an initial ellipsoid that contains the feasible region, and then successively shrinking the ellipsoid until it contains the optimal solution. The algorithm is guaranteed to converge to an optimal solution in a finite number of steps.

The method has a wide range of practical applications in operations research. It can be used to solve linear programming problems, as well as more general convex optimization problems. The method has been applied to a variety of fields, including economics, engineering, and computer science. Some specific applications of the Ellipsoid Method include portfolio optimization, network flow problems, and the design of control systems. The method has also been used to solve problems in combinatorial optimization, such as the traveling salesman problem.

## What is 🪜 Parallel Cut?

In the context of the Ellipsoid Method, a parallel cut refers to a pair of linear constraints of the form aTx <= b and -aTx <= -b, where a is a vector of coefficients and b is a scalar constant. These constraints are said to be parallel because they have the same normal vector a, but opposite signs. When a parallel cut is encountered during the Ellipsoid Method, both constraints can be used simultaneously to generate a new ellipsoid. This can improve the convergence rate of the method, especially for problems with many parallel constraints.

## ✨ Features

- Support parallel cut.
- Support discrete optimization.
- Support traditional or stable version.
- [Modern CMake practices](https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/)
- Suited for single header libraries and projects of any scale
- Clean separation of library and executable code
- Integrated test suite
- Continuous integration via [GitHub Actions](https://help.github.com/en/actions/)
- Code coverage via [codecov](https://codecov.io)
- Code formatting enforced by [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- Reproducible dependency management via [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- Installable target with `find_package` support
- Automatic [documentation](https://thelartians.github.io/ModernCppStarter) and deployment with [Doxygen](https://www.doxygen.nl) and [GitHub Pages](https://pages.github.com)
- Optional [clang-tidy](#additional-tools) static analysis

## Usage

### Adjust the template to your needs

- Use this repo [as a template](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template).
- Replace all occurrences of "EllAlgo" in the relevant CMakeLists.txt with the name of your project
  - Capitalization matters here: `EllAlgo` means the name of the project, while `ellalgo` is used in file names.
  - Remember to rename the `include/ellalgo` directory to use your project's lowercase name and update all relevant `#include`s accordingly.
- Replace the source files with your own
- Add [your project's codecov token](https://docs.codecov.io/docs/quick-start) to your project's github secrets under `CODECOV_TOKEN`
- Happy coding!

Eventually, you can remove any unused files, such as the standalone directory or irrelevant github workflows for your project.
Feel free to replace the License with one suited for your project.

A single `CMakeLists.txt` at the project root defines the library, tests, standalone executable and optional targets.
During development it is usually convenient to [build everything at once](#build-everything-at-once).

### Build and run the standalone target

Use the following commands from the project's root directory to build and run the executable target.

```bash
cmake -B build
cmake --build build
./build/EllAlgo --help
```

### Build and run test suite

Use the following commands from the project's root directory to run the test suite.

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure

# or simply call the executable:
./build/EllAlgoTests
```

To collect code coverage information, run CMake with the `-DELLALGO_ENABLE_COVERAGE=1` option
and build the `coverage` target (GCC/Clang, requires gcovr).

### Run clang-format

Use the following command from the project's root directory to fix C++ source style.
This requires _clang-format_ to be installed on the current system.

```bash
clang-format -i include/ source/ test/source/ standalone/source/ bench/*.cpp
```

### Build the documentation

The documentation is automatically built and [published](https://thelartians.github.io/ModernCppStarter) whenever a [GitHub Release](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) is created.
To manually build documentation, call the following command.

```bash
cmake -B build -DELLALGO_BUILD_DOCS=ON
cmake --build build --target GenerateDocs
# view the docs
open build/doxygen/html/index.html
```

To build the documentation locally, you will need Doxygen and Graphviz installed on your system.

### Build everything at once
\anchor build-everything-at-once

A single build configuration from the project root builds the library, tests and standalone together.
This is useful during development, as it exposes all targets to your IDE and avoids redundant builds of the library.

```bash
cmake -B build
cmake --build build

# run tests
./build/EllAlgoTests
# run standalone
./build/EllAlgo --help
# build docs (requires -DELLALGO_BUILD_DOCS=ON at configure time)
cmake --build build --target GenerateDocs
```

### Additional tools
\anchor additional-tools

The following optional tools can be enabled through CMake configuration arguments.

#### clang-tidy

Static analysis with clang-tidy can be enabled by configuring CMake with `-DELLALGO_ENABLE_CLANG_TIDY=ON`
and building the `clang-tidy` target:

```bash
cmake -B build -DELLALGO_ENABLE_CLANG_TIDY=ON
cmake --build build --target clang-tidy
```

#### Code coverage

Coverage reporting (GCC/Clang, requires gcovr) can be enabled with `-DELLALGO_ENABLE_COVERAGE=1`:

```bash
cmake -B build -DELLALGO_ENABLE_COVERAGE=1
cmake --build build
cmake --build build --target coverage
```

## ❓ FAQ

> Can I use this for header-only libraries?

Yes, however you will need to change the library type to an `INTERFACE` library (empty sources, all `PUBLIC` flags) in [CMakeLists.txt](CMakeLists.txt).
See [here](https://github.com/TheLartians/StaticTypeInfo) for an example header-only library based on the template.

> I don't need a standalone target / documentation. How can I get rid of it?

Simply remove the standalone / documentation directory and according github workflow file.

> Can I build the standalone and tests at the same time? / How can I tell my IDE about all subprojects?

All targets are defined in the single root `CMakeLists.txt`, so a single `cmake -B build` configuration exposes the library, tests and standalone to your IDE at once.

> I see you are using `GLOB` to add source files in CMakeLists.txt. Isn't that evil?

Glob is considered bad because any changes to the source file structure [might not be automatically caught](https://cmake.org/cmake/help/latest/command/file.html#filesystem) by CMake's builders and you will need to manually invoke CMake on changes.
I personally prefer the `GLOB` solution for its simplicity, but feel free to change it to explicitly listing sources.

> I want create additional targets that depend on my library. Should I modify the main CMakeLists to include them?

Avoid including derived projects from the libraries CMakeLists (even though it is a common sight in the C++ world), as this effectively inverts the dependency tree and makes the build system hard to reason about.
Instead, create a new directory or project with a CMakeLists that adds the library as a dependency (e.g. like the [test_installed](test_installed/CMakeLists.txt) directory).
Depending type it might make sense move these components into a separate repositories and reference a specific commit or version of the library.
This has the advantage that individual libraries and components can be improved and updated independently.

> You recommend to add external dependencies using FetchContent. Will this force users of my library to use FetchContent as well?

[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) is part of CMake itself, so users only need a recent CMake version.
If problems do arise, users can always opt-out by defining the CMake or env variable [`FETCHCONTENT_FULLY_DISCONNECTED`](https://cmake.org/cmake/help/latest/module/FetchContent.html#variable:FETCHCONTENT_FULLY_DISCONNECTED) or by providing the dependencies through [`FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>`](https://cmake.org/cmake/help/latest/module/FetchContent.html#variable:FETCHCONTENT_SOURCE_DIR_%3CUPPERCASENAME%3E).
This should also enable users to use the project with their favorite external C++ dependency manager, such as vcpkg or Conan.

> Can I configure and build my project offline?

No internet connection is required for building the project, however when using FetchContent missing dependencies are downloaded at configure time.
To avoid redundant downloads, it's highly recommended to point `FETCHCONTENT_SOURCE_DIR_*` at local checkouts of the dependencies (e.g. `FETCHCONTENT_SOURCE_DIR_SPDLOG=$HOME/git/spdlog`).
This will enable offline configurations when dependencies are already available locally.

> Can I use CPack to create a package installer for my project?

As there are a lot of possible options and configurations, this is not (yet) in the scope of this template. See the [CPack documentation](https://cmake.org/cmake/help/latest/module/CPack.html) for more information on setting up CPack installers.

> This is too much, I just want to play with C++ code and test some libraries.

Perhaps the [MiniCppStarter](https://github.com/TheLartians/MiniCppStarter) is something for you!

## Related projects and alternatives

- [**ellalgo-simple**](https://github.com/luk036/ellalgo-simple): Simplified header-only variant (xmake/CMake, lighter CI)
- [**ModernCppStarter & PVS-Studio Static Code Analyzer**](https://github.com/viva64/pvs-studio-cmake-examples/tree/master/modern-cpp-starter): Official instructions on how to use the ModernCppStarter with the PVS-Studio Static Code Analyzer.
- [**cpp-best-practices/gui_starter_template**](https://github.com/cpp-best-practices/gui_starter_template/): A popular C++ starter project, created in 2017.
- [**filipdutescu/modern-cpp-template**](https://github.com/filipdutescu/modern-cpp-template): A recent starter using a more traditional approach for CMake structure and dependency management.
- [**vector-of-bool/pitchfork**](https://github.com/vector-of-bool/pitchfork/): Pitchfork is a Set of C++ Project Conventions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=luk036/ellalgo-cpp,cpp-best-practices/gui_starter_template,filipdutescu/modern-cpp-template&type=Date)](https://star-history.com/#luk036/ellalgo-cpp&cpp-best-practices/gui_starter_template&filipdutescu/modern-cpp-template&Date)
