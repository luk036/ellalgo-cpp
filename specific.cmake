find_package(OpenBLAS REQUIRED)
if (OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  # target_include_directories(OpenBLAS::OpenBLAS SYSTEM INTERFACE ${OpenBLAS_INCLUDE_DIRS})
endif (OpenBLAS_FOUND)

CPMAddPackage("gh:xtensor-stack/xtl#0.7.2")
CPMAddPackage("gh:xtensor-stack/xtensor#0.21.0")
CPMAddPackage("gh:xtensor-stack/xtensor-blas#0.18.0")
