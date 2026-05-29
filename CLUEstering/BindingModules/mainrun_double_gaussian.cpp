#include <alpaka/alpaka.hpp>
#include "Run.hpp"

namespace ALPAKA_BACKEND {

template void mainRun<double, clue::GaussianKernel>(
    double, double, double, double, int,
    std::vector<uint8_t>,
    py::array_t<double>, py::array_t<int>,
    const clue::GaussianKernel<double>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<double>&);

void register_mainRun_double_GaussianKernel(py::module_& m) {
  m.def("mainRun", &mainRun<double, clue::GaussianKernel>);
}

}  // namespace ALPAKA_BACKEND
