#include <alpaka/alpaka.hpp>
#include "Run.hpp"

namespace ALPAKA_BACKEND {

template void mainRun<float, clue::FlatKernel>(
    float, float, float, float, int,
    std::vector<uint8_t>,
    py::array_t<float>, py::array_t<int>,
    const clue::FlatKernel<float>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<float>&);

void register_mainRun_float_FlatKernel(py::module_& m) {
  m.def("mainRun", &mainRun<float, clue::FlatKernel>);
}

}  // namespace ALPAKA_BACKEND
