cmake_minimum_required(VERSION 3.14)
project(
  EllAlgo
  VERSION 1.6.10
  LANGUAGES CXX
)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# ---------------------------------------------------------------------------
# Compiler warnings (as errors)
# ---------------------------------------------------------------------------

function(target_enable_warnings TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /W4 /WX /utf-8 /permissive- /wd4819)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ELLALGO_BUILD_TESTS "Build tests" ON)
option(ELLALGO_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ELLALGO_BUILD_DOCS "Build Doxygen documentation" OFF)
option(ELLALGO_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" OFF)
option(ELLALGO_ENABLE_COVERAGE "Enable coverage reporting" OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# spdlog (bundles fmt) - used by the library's logger. Prefer a consumer- or system-provided spdlog
# (e.g. system install, vcpkg/Conan, or a CPM dependency); only fall back to FetchContent when no
# spdlog target is available.
find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND AND NOT TARGET spdlog::spdlog)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
  )
  set(SPDLOG_INSTALL OFF)
  FetchContent_MakeAvailable(spdlog)
endif()

# Only the optional logger uses spdlog. The library is built against its headers but does not link
# it, keeping the installed package's exported link interface free of spdlog; executables that use
# the logger link spdlog explicitly. The compile definitions (SPDLOG_COMPILED_LIB) are propagated so
# the library's logger shares the same compiled spdlog registry as the executables.
if(TARGET spdlog::spdlog)
  get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(SPDLOG_COMPILE_DEFS spdlog::spdlog INTERFACE_COMPILE_DEFINITIONS)
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
file(GLOB_RECURSE ELLALGO_SOURCES CONFIGURE_DEPENDS "source/*.cpp")
add_library(EllAlgo ${ELLALGO_SOURCES})
add_library(EllAlgo::EllAlgo ALIAS EllAlgo)

target_include_directories(
  EllAlgo PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>
)
# keep spdlog headers out of the -Werror scope (MSVC C4251 etc. in third-party headers)
target_include_directories(EllAlgo SYSTEM PRIVATE ${SPDLOG_INCLUDE_DIRS})
target_compile_definitions(EllAlgo PRIVATE ${SPDLOG_COMPILE_DEFS})
target_enable_warnings(EllAlgo)

# ---------------------------------------------------------------------------
# Standalone executable
# ---------------------------------------------------------------------------

# cxxopts is a header-only CLI parser used by the standalone example. Prefer a consumer- or
# system-provided cxxopts (e.g. vcpkg/Conan or a CPM dependency); only fall back to FetchContent
# when no cxxopts target is available.
find_package(cxxopts CONFIG QUIET)
if(NOT cxxopts_FOUND AND NOT TARGET cxxopts::cxxopts)
  set(CXXOPTS_BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE
  )
  set(CXXOPTS_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE
  )
  set(CXXOPTS_ENABLE_INSTALL
      OFF
      CACHE BOOL "" FORCE
  )
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.1
  )
  FetchContent_MakeAvailable(cxxopts)
endif()
# keep third-party headers out of our -Werror scope
if(TARGET cxxopts::cxxopts)
  get_target_property(CXXOPTS_INCLUDE_DIRS cxxopts::cxxopts INTERFACE_INCLUDE_DIRECTORIES)
endif()

file(GLOB ELLALGO_STANDALONE_SOURCES CONFIGURE_DEPENDS "standalone/source/*.cpp")
add_executable(EllAlgoStandalone ${ELLALGO_STANDALONE_SOURCES})
set_target_properties(EllAlgoStandalone PROPERTIES OUTPUT_NAME "EllAlgo")
target_include_directories(EllAlgoStandalone SYSTEM PRIVATE ${CXXOPTS_INCLUDE_DIRS})
target_link_libraries(EllAlgoStandalone PRIVATE EllAlgo cxxopts::cxxopts spdlog::spdlog)
target_enable_warnings(EllAlgoStandalone)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

if(ELLALGO_BUILD_TESTS)
  # Try system-installed doctest first (Ubuntu: doctest-dev, macOS: brew install doctest, Termux:
  # doctest)
  find_package(doctest CONFIG QUIET)

  if(doctest_FOUND)
    message(STATUS "Found system doctest: ${doctest_DIR}")
    # Create a compatibility target so downstream uses 'doctest' consistently
    if(NOT TARGET doctest)
      add_library(doctest INTERFACE)
      target_link_libraries(doctest INTERFACE doctest::doctest)
    endif()
  else()
    set(DOCTEST_VERSION 2.4.11)
    set(DOCTEST_DIR "${CMAKE_BINARY_DIR}/_deps/doctest")
    set(DOCTEST_HEADER "${DOCTEST_DIR}/doctest/doctest.h")

    if(NOT EXISTS "${DOCTEST_HEADER}")
      file(MAKE_DIRECTORY "${DOCTEST_DIR}/doctest")
      file(DOWNLOAD
           "https://raw.githubusercontent.com/doctest/doctest/v${DOCTEST_VERSION}/doctest/doctest.h"
           "${DOCTEST_HEADER}" STATUS download_status
      )
      list(GET download_status 0 status_code)
      if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download doctest (status ${status_code})")
      endif()
    endif()

    add_library(doctest INTERFACE)
    target_include_directories(doctest SYSTEM INTERFACE "${DOCTEST_DIR}")
    if(MSVC)
      target_compile_options(doctest INTERFACE /wd5285)
    endif()
  endif()

  # Code formatting targets (format / fix-format) via Format.cmake
  FetchContent_Declare(
    format
    GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
    GIT_TAG v1.7.3
  )
  FetchContent_MakeAvailable(format)

  file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "test/source/*.cpp")
  add_executable(EllAlgoTests ${TEST_SOURCES})
  target_link_libraries(EllAlgoTests PRIVATE EllAlgo doctest spdlog::spdlog)
  # keep spdlog headers out of the -Werror scope (MSVC C4251 etc. in third-party headers)
  target_include_directories(EllAlgoTests SYSTEM PRIVATE ${SPDLOG_INCLUDE_DIRS})
  target_enable_warnings(EllAlgoTests)
  add_test(NAME EllAlgoTests COMMAND EllAlgoTests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks (using nanobench, header-only)
# ---------------------------------------------------------------------------

if(ELLALGO_BUILD_BENCHMARKS)
  FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
  )
  FetchContent_MakeAvailable(nanobench)
  get_target_property(NANOBENCH_INCLUDE_DIRS nanobench INTERFACE_INCLUDE_DIRECTORIES)

  file(GLOB_RECURSE ALL_BENCH_CPP CONFIGURE_DEPENDS "bench/*.cpp")
  foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
    get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

    # Avoid name collision
    set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

    add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${NANOBENCH_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE nanobench EllAlgo spdlog::spdlog)
    target_enable_warnings(${TARGET_NAME})

    add_test(NAME ${ONE_BENCH_EXEC} COMMAND ${TARGET_NAME})
  endforeach()
endif()

# ---------------------------------------------------------------------------
# clang-tidy
# ---------------------------------------------------------------------------

if(ELLALGO_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  if(CLANG_TIDY)
    add_custom_target(
      clang-tidy
      COMMAND
        ${CLANG_TIDY} --extra-arg=-std=c++20 --extra-arg=-I${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ellalgo/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ellalgo/oracles/*.hpp
      COMMENT "Running clang-tidy static analysis"
    )
  else()
    message(WARNING "clang-tidy not found - analysis target unavailable")
  endif()
endif()

# ---------------------------------------------------------------------------
# Doxygen documentation
# ---------------------------------------------------------------------------

if(ELLALGO_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME ${PROJECT_NAME})
    set(DOXYGEN_PROJECT_VERSION ${PROJECT_VERSION})
    set(DOXYGEN_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

    configure_file(documentation/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    add_custom_target(
      GenerateDocs
      ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ellalgo.svg"
              "${DOXYGEN_OUTPUT_DIRECTORY}/"
      COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
      COMMENT "Generating Doxygen HTML documentation in ${DOXYGEN_OUTPUT_DIRECTORY}/html"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
  else()
    add_custom_target(
      GenerateDocs COMMAND ${CMAKE_COMMAND} -E echo
                           "Doxygen not found - skipping documentation generation"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# Coverage report target (GCC/Clang only - requires gcovr)
# ---------------------------------------------------------------------------

if(ELLALGO_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  # Every target that links the instrumented library also needs the coverage runtime at link time.
  foreach(COVERAGE_TARGET EllAlgo EllAlgoTests EllAlgoStandalone)
    target_compile_options(${COVERAGE_TARGET} PRIVATE -O0 -g -fprofile-arcs -ftest-coverage)
  endforeach()
  foreach(COVERAGE_EXEC EllAlgoTests EllAlgoStandalone)
    target_link_options(${COVERAGE_EXEC} PRIVATE -fprofile-arcs -ftest-coverage)
  endforeach()

  find_program(GCOVR_PATH gcovr)
  if(GCOVR_PATH)
    set(extra_gcovr_args "")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
      find_program(LLVM_COV_PATH llvm-cov)
      if(LLVM_COV_PATH)
        list(APPEND extra_gcovr_args --llvm-cov-binary ${LLVM_COV_PATH})
      endif()
    endif()
    add_custom_target(
      coverage
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
      COMMAND
        ${GCOVR_PATH} --root ${PROJECT_SOURCE_DIR} --html-details
        ${CMAKE_BINARY_DIR}/coverage/index.html --exclude test/ --exclude build/ ${extra_gcovr_args}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating coverage report in ${CMAKE_BINARY_DIR}/coverage/index.html"
    )
  else()
    message(
      WARNING "gcovr not found - coverage target unavailable. Install with: pip install gcovr"
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/EllAlgoConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/EllAlgoConfig.cmake"
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/EllAlgoTargets.cmake\")\n"
)

install(
  TARGETS EllAlgo
  EXPORT EllAlgoTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  EXPORT EllAlgoTargets
  FILE EllAlgoTargets.cmake
  NAMESPACE EllAlgo::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EllAlgo
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/EllAlgoConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/EllAlgoConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EllAlgo
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
