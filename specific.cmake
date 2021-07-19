CPMAddPackage("gh:xtensor-stack/xtl#0.6.23")
CPMAddPackage("gh:xtensor-stack/xtensor#0.22.0")

find_package(OpenBLAS REQUIRED)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
  # target_include_directories(OpenBLAS::OpenBLAS SYSTEM INTERFACE ${OpenBLAS_INCLUDE_DIRS})
endif(OpenBLAS_FOUND)

find_package(xtensor-blas REQUIRED)
if(xtensor-blas_FOUND)
  message(STATUS "Found xtensor-blas: ${xtensor_blas_INCLUDE_DIRS}")
  include_directories(${xtensor_blas_INCLUDE_DIRS})
else()
  message(STATUS "Something wrong with xtensor-blas")
  include_directories(${xtensor_blas_INCLUDE_DIRS})
endif(xtensor-blas_FOUND)
# remember to turn off the warning
