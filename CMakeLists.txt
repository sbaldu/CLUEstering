cmake_minimum_required(VERSION 3.16.0)
project(CLUEstering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_definitions(DEBUG)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-O2")
endif()

# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")
add_subdirectory(extern/pybind11)

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  alpaka
  GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
  GIT_TAG develop
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(alpaka)

if(NOT DEFINED CPU_ONLY)
  set(CPU_ONLY OFF)
endif()
if(NOT DEFINED SERIAL_ONLY)
  set(SERIAL_ONLY OFF)
endif()

include_directories(include)

# create lib directory in CLUEstering folder
execute_process(COMMAND mkdir -p ./CLUEstering/lib)

add_subdirectory(CLUEstering/BindingModules)
