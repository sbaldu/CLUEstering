#pragma once

#include "Run.hpp"

// Per-(TInput, Kernel) explicit instantiations live in mainrun_*.cpp.
// Declaring them extern here prevents binding_*.cpp files from generating their
// own copies, so those TUs only compile the module-registration glue.
namespace ALPAKA_BACKEND {

extern template void mainRun<float, clue::FlatKernel>(
    float, float, float, float, int,
    std::vector<uint8_t>,
    py::array_t<float>, py::array_t<int>,
    const clue::FlatKernel<float>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<float>&);

extern template void mainRun<float, clue::ExponentialKernel>(
    float, float, float, float, int,
    std::vector<uint8_t>,
    py::array_t<float>, py::array_t<int>,
    const clue::ExponentialKernel<float>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<float>&);

extern template void mainRun<float, clue::GaussianKernel>(
    float, float, float, float, int,
    std::vector<uint8_t>,
    py::array_t<float>, py::array_t<int>,
    const clue::GaussianKernel<float>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<float>&);

extern template void mainRun<double, clue::FlatKernel>(
    double, double, double, double, int,
    std::vector<uint8_t>,
    py::array_t<double>, py::array_t<int>,
    const clue::FlatKernel<double>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<double>&);

extern template void mainRun<double, clue::ExponentialKernel>(
    double, double, double, double, int,
    std::vector<uint8_t>,
    py::array_t<double>, py::array_t<int>,
    const clue::ExponentialKernel<double>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<double>&);

extern template void mainRun<double, clue::GaussianKernel>(
    double, double, double, double, int,
    std::vector<uint8_t>,
    py::array_t<double>, py::array_t<int>,
    const clue::GaussianKernel<double>&,
    int,
    std::optional<py::array_t<uint32_t>>,
    int32_t, std::size_t, std::size_t,
    const clue::internal::MetricDescriptor<double>&);

// Forward declarations for per-(TInput, Kernel) registration helpers.
// Each is defined in the corresponding mainrun_*.cpp TU so that the pybind11
// wrapper template for each overload is also instantiated there, not in the
// binding_*.cpp file.
void register_mainRun_float_FlatKernel(py::module_& m);
void register_mainRun_float_ExponentialKernel(py::module_& m);
void register_mainRun_float_GaussianKernel(py::module_& m);
void register_mainRun_double_FlatKernel(py::module_& m);
void register_mainRun_double_ExponentialKernel(py::module_& m);
void register_mainRun_double_GaussianKernel(py::module_& m);

}  // namespace ALPAKA_BACKEND
