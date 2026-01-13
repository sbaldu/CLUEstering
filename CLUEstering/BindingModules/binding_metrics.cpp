
#include "MetricTags.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(CLUE_DistanceMetrics, m) {
  m.doc() = "Binding of the distance metrics used in the CLUE algorithm.";

  pybind11::class_<clue::EuclideanMetricTag>(m, "EuclideanMetric").def(pybind11::init<>());
  pybind11::class_<clue::WeightedEuclideanTag>(m, "WeightedEuclideanMetric")
      .def(pybind11::init<std::vector<float>>());
  pybind11::class_<clue::PeriodicEuclideanTag>(m, "PeriodicEuclideanMetric")
      .def(pybind11::init<std::vector<float>>());
  pybind11::class_<clue::ManhattanTag>(m, "ManhattanMetric").def(pybind11::init<>());
  pybind11::class_<clue::ChebyshevTag>(m, "ChebyshevMetric").def(pybind11::init<>());
  pybind11::class_<clue::WeightedChebyshevTag>(m, "WeightedChebyshevMetric")
      .def(pybind11::init<std::vector<float>>());
}
