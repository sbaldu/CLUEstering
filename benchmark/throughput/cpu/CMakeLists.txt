

add_library(serial SHARED ${CMAKE_SOURCE_DIR}/header.cpp)
target_compile_definitions(serial PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  target_include_directories(
    serial PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                       ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})

