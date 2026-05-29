
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

namespace alpaka_tbb_async {

  PYBIND11_MODULE(CLUE_CPU_TBB, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with TBB";

    m.def("listDevices",
          &alpaka_tbb_async::listDevices,
          "List the available devices for the TBB backend");

    register_mainRun_float_FlatKernel(m);
    register_mainRun_float_ExponentialKernel(m);
    register_mainRun_float_GaussianKernel(m);
    register_mainRun_double_FlatKernel(m);
    register_mainRun_double_ExponentialKernel(m);
    register_mainRun_double_GaussianKernel(m);
  }
};  // namespace alpaka_tbb_async
