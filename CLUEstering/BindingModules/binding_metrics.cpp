
#include "MetricTags.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(CLUE_DistanceMetrics, m) {
  m.doc() = "Binding of the distance metrics used in the CLUE algorithm.";

  pybind11::class_<clue::DistanceMetricTag>(m, "DistanceMetricTag").def(pybind11::init<>());
  pybind11::class_<clue::EuclideanMetricTag, clue::DistanceMetricTag>(m, "EuclideanMetric")
      .def(pybind11::init<>());
  pybind11::class_<clue::WeightedEuclideanTag, clue::DistanceMetricTag>(m,
                                                                        "WeightedEuclideanMetric")
      .def(pybind11::init<std::vector<float>>());
  pybind11::class_<clue::PeriodicEuclideanTag, clue::DistanceMetricTag>(m,
                                                                        "PeriodicEuclideanMetric")
      .def(pybind11::init<std::vector<float>>());
  pybind11::class_<clue::ManhattanTag, clue::DistanceMetricTag>(m, "ManhattanMetric")
      .def(pybind11::init<>());
  pybind11::class_<clue::ChebyshevTag, clue::DistanceMetricTag>(m, "ChebyshevMetric")
      .def(pybind11::init<>());
  pybind11::class_<clue::WeightedChebyshevTag, clue::DistanceMetricTag>(m,
                                                                        "WeightedChebyshevMetric")
      .def(pybind11::init<std::vector<float>>());
}
