
#include <vector>

#include "include/Clustering.h"
#include "include/Kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

using run_t = std::function<std::vector<std::vector<int>>(float,
                                                          float,
                                                          float,
                                                          int,
                                                          const std::vector<domain_t> &,
                                                          const kernel &,
                                                          const std::vector<std::vector<float>> &,
                                                          const std::vector<float> &)>;

template <uint16_t n_dim>
std::unordered_map<uint16_t, run_t> generate_run_map(
    std::unordered_map<uint16_t, run_t>& run_map) {
  if constexpr (n_dim == 0) {
    return run_map;
  } else {
    auto run_function = [](float dc,
                           float rhoc,
                           float odf,
                           int ppbin,
                           const std::vector<domain_t> &domains,
                           const kernel& ker,
                           const std::vector<std::vector<float>>& coords,
                           const std::vector<float>& weight) {
      ClusteringAlgo<n_dim> algo(dc, rhoc, odf, ppbin, domains);
      algo.setPoints(coords[0].size(), coords, weight);

      return algo.makeClusters(ker);
    };
    run_map[n_dim] = run_function;

    return generate_run_map<n_dim - 1>(run_map);
  }
}

template <uint16_t n_dim>
std::unordered_map<uint16_t, run_t> generate_run_map() {
  std::unordered_map<uint16_t, run_t> run_map;
  auto run_function = [](float dc,
                         float rhoc,
                         float odf,
                         int ppbin,
                         const std::vector<domain_t> &domains,
                         const kernel &ker,
                         const std::vector<std::vector<float>> &coords,
                         const std::vector<float> &weight) {
    ClusteringAlgo<n_dim> algo(dc, rhoc, odf, ppbin, domains);
    algo.setPoints(coords[0].size(), coords, weight);

    return algo.makeClusters(ker);
  };
  run_map[n_dim] = run_function;

  return generate_run_map<n_dim - 1>(run_map);
}

std::unordered_map<uint16_t, run_t> run_map = generate_run_map<20>();

std::vector<std::vector<int>> mainRun(float dc,
                                      float rhoc,
                                      float outlier,
                                      int pPBin,
                                      std::vector<domain_t> domains,
                                      const kernel& ker,
                                      const std::vector<std::vector<float>>& coords,
                                      const std::vector<float>& weight,
                                      int Ndim) {
  // Running the clustering algorithm //
  return run_map[Ndim](dc, rhoc, outlier, pPBin, domains, ker, coords, weight);
}


//////////////////////////////
//////  Binding module  //////
//////////////////////////////
PYBIND11_MODULE(CLUEsteringCPP, m) {
  m.doc() = "Binding for CLUE";

  pybind11::class_<domain_t>(m, "domain_t").def(pybind11::init<>()).def(pybind11::init<float, float>());

  pybind11::class_<kernel>(m, "kernel").def(pybind11::init<>()).def("operator()", &kernel::operator());
  pybind11::class_<flatKernel, kernel>(m, "flatKernel")
      .def(pybind11::init<float>())
      .def("operator()", &flatKernel::operator());
  pybind11::class_<gaussianKernel, kernel>(m, "gaussianKernel")
      .def(pybind11::init<float, float, float>())
      .def("operator()", &gaussianKernel::operator());
  pybind11::class_<exponentialKernel, kernel>(m, "exponentialKernel")
      .def(pybind11::init<float, float>())
      .def("operator()", &exponentialKernel::operator());
  pybind11::class_<customKernel, kernel>(m, "customKernel")
      .def(pybind11::init<kernel_t>())
      .def("operator()", &customKernel::operator());

  m.def("mainRun", &mainRun, "mainRun");
}
