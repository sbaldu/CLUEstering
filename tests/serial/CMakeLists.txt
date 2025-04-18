
add_executable(serial.out ../test_tiles.cpp)
target_include_directories(
  serial.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include/CLUEstering/ ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

