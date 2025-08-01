set(CMAKE_CXX_FLAGS
    "-O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
add_executable(sycl_cpu.out ../main.cpp)
target_include_directories(
  sycl_cpu.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../../include
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_include_directories(sycl_cpu.out PRIVATE ${oneDPL_DIR})
target_compile_options(sycl_cpu.out PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                            -DALPAKA_SYCL_ONEAPI_CPU -fsycl)
target_link_libraries(sycl_cpu.out PRIVATE -fsycl)

add_executable(sycl_gpu.out ../main.cpp)
target_include_directories(
  sycl_gpu.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../../include
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_include_directories(sycl_gpu.out PRIVATE ${oneDPL_DIR})
target_compile_options(sycl_gpu.out PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                            -DALPAKA_SYCL_ONEAPI_GPU -fsycl)
target_link_libraries(sycl_gpu.out PRIVATE -fsycl)
