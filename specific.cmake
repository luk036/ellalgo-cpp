CPMAddPackage("gh:xtensor-stack/xtl#0.7.2")
CPMAddPackage("gh:xtensor-stack/xtensor#0.23.10")

find_package(xtensor-blas REQUIRED)
if (xtensor-blas_FOUND)
  message(STATUS "Found xtensor-blas: ${xtensor-blas_INCLUDE_DIRS}")
  include_directories(${xtensor-blas_INCLUDE_DIRS})
endif(xtensor-blas_FOUND)

find_package(OpenBLAS REQUIRED)
if (OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  # target_include_directories(OpenBLAS::OpenBLAS SYSTEM INTERFACE ${OpenBLAS_INCLUDE_DIRS})
endif (OpenBLAS_FOUND)
