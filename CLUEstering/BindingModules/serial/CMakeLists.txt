set(CLUE_MAINRUN_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_float_flat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_float_exponential.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_float_gaussian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_double_flat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_double_exponential.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../mainrun_double_gaussian.cpp)

pybind11_add_module(CLUE_CPU_Serial NO_EXTRAS SHARED binding_cpu.cpp ${CLUE_MAINRUN_SRCS})
target_include_directories(CLUE_CPU_Serial
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(CLUE_CPU_Serial PRIVATE alpaka::alpaka Boost::atomic)
target_compile_definitions(
  CLUE_CPU_Serial PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
target_precompile_headers(CLUE_CPU_Serial REUSE_FROM CLUE_Convolutional_Kernels)
set_target_properties(
  CLUE_CPU_Serial
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
install(TARGETS CLUE_CPU_Serial LIBRARY DESTINATION CLUEstering/lib)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_Serial
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CLUE_CPU_Serial>
          ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
