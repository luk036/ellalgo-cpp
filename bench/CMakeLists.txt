cmake_minimum_required(VERSION 3.14...3.22)

project(EllAlgoBenchmarks LANGUAGES CXX)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)
include(../cmake/specific.cmake)

# nanobench is a header-only library for microbenchmarking
CPMAddPackage(
  NAME nanobench
  GITHUB_REPOSITORY martinus/nanobench
  GIT_TAG v4.3.11
)

CPMAddPackage(NAME EllAlgo SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# ---- Create binary ----

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

enable_testing()

# enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(EllAlgo PUBLIC -Wall -Wextra)
elseif(MSVC)
  target_compile_options(EllAlgo PUBLIC /utf-8 /W4 /WX)
endif()

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})

  get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

  # Avoid name collision
  set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

  add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
  set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)

  target_link_libraries(${TARGET_NAME} nanobench EllAlgo::EllAlgo ${SPECIFIC_LIBS})

  add_test(NAME ${ONE_BENCH_EXEC} COMMAND ${TARGET_NAME})
endforeach()
