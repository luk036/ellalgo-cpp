cmake_minimum_required(VERSION 3.14)
project(EllAlgoInstalledTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Verify that the installed package can be found and linked via find_package.
find_package(EllAlgo REQUIRED)

add_executable(ellalgo_installed_test main.cpp)
target_link_libraries(ellalgo_installed_test PRIVATE EllAlgo::EllAlgo)

# Verify the dependency-free cutting-plane framework component links standalone.
add_executable(ellalgo_core_installed_test main_core.cpp)
target_link_libraries(ellalgo_core_installed_test PRIVATE EllAlgo::EllAlgoCore)

enable_testing()
add_test(NAME ellalgo_installed_test COMMAND ellalgo_installed_test)
add_test(NAME ellalgo_core_installed_test COMMAND ellalgo_core_installed_test)
