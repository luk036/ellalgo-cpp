cmake_minimum_required(VERSION 3.14...3.22)

project(EllAlgoBenchmarks LANGUAGES CXX)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)
include(../cmake/specific.cmake)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.8.3
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

if(benchmark_ADDED)
  # enable c++11 to avoid compilation errors
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 14)
endif()

CPMAddPackage(NAME EllAlgo SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# ---- Create binary ----

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

enable_testing()

# enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(EllAlgo PUBLIC -Wall -Wextra)
elseif(MSVC)
  target_compile_options(EllAlgo PUBLIC /W4 /WX /wd4819 /wd4146 /wd4127 /wd4996)
  # target_compile_definitions(EllAlgoTests PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
endif()

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})

  get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

  # Avoid name collision
  set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

  add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
  set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 14)

  target_link_libraries(${TARGET_NAME} benchmark::benchmark EllAlgo::EllAlgo ${SPECIFIC_LIBS})

  # add_test(${TARGET_NAME} ${ONE_BENCH_EXEC})
  add_test(NAME ${ONE_BENCH_EXEC} COMMAND ${TARGET_NAME})
endforeach()
