cmake_minimum_required(VERSION 3.14)
project(
  EllAlgo
  VERSION 1.6.8
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# Compiler warnings (as errors)
# ---------------------------------------------------------------------------

function(target_enable_warnings TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /W4 /WX /wd4819 /utf-8)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(ELLALGO_BUILD_TESTS "Build tests" ON)
option(ELLALGO_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ELLALGO_BUILD_STANDALONE "Build standalone executable" OFF)
option(ELLALGO_BUILD_DOCS "Build Doxygen documentation" OFF)
option(ELLALGO_ENABLE_COVERAGE "Enable coverage reporting" OFF)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------

file(GLOB_RECURSE ELLALGO_SOURCES "source/*.cpp")
add_library(ellalgo ${ELLALGO_SOURCES})
target_include_directories(
  ellalgo PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_enable_warnings(ellalgo)

# spdlog dependency (used by logger)
include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.17.0
)
set(SPDLOG_INSTALL ON)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(ellalgo PRIVATE spdlog::spdlog)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

if(ELLALGO_BUILD_TESTS)
  # Try system-installed doctest first (Ubuntu: doctest-dev, macOS: brew install doctest, Termux: doctest)
  find_package(doctest CONFIG QUIET)

  if(doctest_FOUND)
    message(STATUS "Found system doctest: ${doctest_DIR}")
    if(NOT TARGET doctest)
      add_library(doctest INTERFACE)
      target_link_libraries(doctest INTERFACE doctest::doctest)
    endif()
  else()
    set(DOCTEST_VERSION 2.5.2)
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

  file(GLOB_RECURSE TEST_SOURCES "test/source/*.cpp")
  add_executable(test_all ${TEST_SOURCES})
  target_link_libraries(test_all PRIVATE ellalgo doctest spdlog::spdlog)
  target_enable_warnings(test_all)
  if(MSVC)
    target_compile_options(test_all PRIVATE /utf-8)
  endif()
  add_test(NAME test_all COMMAND test_all)
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

  file(GLOB ALL_BENCH_CPP "bench/*.cpp")
  foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
    get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
    set(TARGET_NAME bench_${ONE_BENCH_EXEC})
    add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
    target_link_libraries(${TARGET_NAME} nanobench ellalgo)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)
    target_enable_warnings(${TARGET_NAME})
    add_test(NAME ${ONE_BENCH_EXEC} COMMAND ${TARGET_NAME})
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Standalone executable
# ---------------------------------------------------------------------------

if(ELLALGO_BUILD_STANDALONE)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.1
  )
  set(CXXOPTS_BUILD_EXAMPLES NO)
  set(CXXOPTS_BUILD_TESTS NO)
  set(CXXOPTS_ENABLE_INSTALL YES)
  FetchContent_MakeAvailable(cxxopts)

  file(GLOB STANDALONE_SOURCES "standalone/source/*.cpp")
  add_executable(ellalgo_standalone ${STANDALONE_SOURCES})
  target_link_libraries(ellalgo_standalone PRIVATE ellalgo cxxopts::cxxopts)
endif()

# ---------------------------------------------------------------------------
# Doxygen documentation
# ---------------------------------------------------------------------------

if(ELLALGO_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    add_custom_target(
      doxygen
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating Doxygen HTML documentation in docs/doxygen/"
    )
  else()
    message(WARNING "Doxygen not found — docs target unavailable. Install with: pip install doxygen")
  endif()
endif()

# ---------------------------------------------------------------------------
# Coverage report target (GCC/Clang only — requires gcovr)
# ---------------------------------------------------------------------------

if(ELLALGO_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
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
      COMMAND ${GCOVR_PATH} --root ${PROJECT_SOURCE_DIR} --html-details
              ${CMAKE_BINARY_DIR}/coverage/index.html --exclude test/ ${extra_gcovr_args}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating coverage report in ${CMAKE_BINARY_DIR}/coverage/index.html"
    )
  else()
    message(WARNING "gcovr not found — coverage target unavailable. Install with: pip install gcovr")
  endif()
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)
