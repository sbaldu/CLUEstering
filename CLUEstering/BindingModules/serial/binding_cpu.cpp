
#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "../RunInstantiations.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace alpaka_serial_sync {

  PYBIND11_MODULE(CLUE_CPU_Serial, m) {
    m.doc() = "Binding of the CLUE algorithm running serially on CPU";

    m.def("listDevices",
          &alpaka_serial_sync::listDevices,
          "List the available devices for the CPU serial backend");

    register_mainRun_float_FlatKernel(m);
    register_mainRun_float_ExponentialKernel(m);
    register_mainRun_float_GaussianKernel(m);
    register_mainRun_double_FlatKernel(m);
    register_mainRun_double_ExponentialKernel(m);
    register_mainRun_double_GaussianKernel(m);
  }
};  // namespace alpaka_serial_sync
