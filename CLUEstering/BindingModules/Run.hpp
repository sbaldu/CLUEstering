
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "MetricTags.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <std::size_t Ndim,
          clue::concepts::convolutional_kernel Kernel,
          clue::concepts::distance_metric<Ndim> DistanceMetric>
void run(float dc,
         float rhoc,
         float dm,
         float seed_dc,
         int pPBin,
         std::vector<uint8_t>&& wrapped,
         std::tuple<float*, int*>&& pData,
         int32_t n_points,
         const Kernel& kernel,
         const DistanceMetric& metric,
         clue::Queue queue,
         size_t block_size) {
  clue::Clusterer<Ndim> algo(queue, dc, rhoc, dm, seed_dc, pPBin);
  algo.setWrappedCoordinates(std::move(wrapped));

  clue::PointsHost<Ndim> h_points(queue, n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim> d_points(queue, n_points);

  algo.make_clusters(queue, h_points, d_points, metric, kernel, block_size);
}

namespace ALPAKA_BACKEND {

  void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs(clue::Platform{});
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend.\n";
      return;
    } else {
      std::cout << backend << " devices found: \n";
      for (auto i = 0u; i < devices.size(); ++i) {
        std::cout << tab << "device " << i << ": " << alpaka::getName(devices[i]) << '\n';
      }
    }
  }

  auto metric_dispatch(const auto& tag) {
    if (auto* m = dynamic_cast<const clue::EuclideanMetricTag*>(&tag)) {
      return clue::EuclideanMetricTag{};
    } else if (auto* m = dynamic_cast<const clue::WeightedEuclideanTag*>(&tag)) {
      return dynamic_cast<const clue::WeightedEuclideanTag&>(tag);
    } else if (auto* m = dynamic_cast<const clue::PeriodicEuclideanTag*>(&tag)) {
      return dynamic_cast<const clue::PeriodicEuclideanTag&>(tag);
    } else if (auto* m = dynamic_cast<const clue::ManhattanTag*>(&tag)) {
      return clue::ManhattanTag{};
    } else if (auto* m = dynamic_cast<const clue::ChebyshevTag*>(&tag)) {
      return clue::ChebyshevTag{};
    } else if (auto* m = dynamic_cast<const clue::WeightedChebyshevTag*>(&tag)) {
      return dynamic_cast<const clue::WeightedChebyshevTag&>(tag);
    }
  }

  template <clue::concepts::convolutional_kernel Kernel>
  void mainRun(float dc,
               float rhoc,
               float dm,
               float seed_dc,
               int pPBin,
               std::vector<uint8_t> wrapped,
               py::array_t<float> data,
               py::array_t<int> results,
               const Kernel& kernel,
               const clue::DistanceMetricTag& metric,
               int Ndim,
               int32_t n_points,
               size_t block_size,
               size_t device_id) {
    auto rData = data.request();
    auto* pData = static_cast<float*>(rData.ptr);
    auto rResults = results.request();
    auto* pResults = static_cast<int*>(rResults.ptr);

    auto queue = clue::get_queue(device_id);

    auto tag_to_metric = []<std::size_t Ndim>(const auto& tag) {
      const auto& metric_tag = metric_dispatch(tag);
      return static_cast<typename clue::internal::TagToMetric<
          std::decay_t<decltype(metric_tag)>>::template type<Ndim> const&>(metric_tag);
    };

    switch (Ndim) {
      [[unlikely]] case (1):
        run<1, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<1>(metric),
                       queue,
                       block_size);
        return;
      [[likely]] case (2):
        run<2, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<2>(metric),
                       queue,
                       block_size);
        return;
      [[likely]] case (3):
        run<3, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<3>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (4):
        run<4, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<4>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (5):
        run<5, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<5>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (6):
        run<6, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<6>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (7):
        run<7, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<7>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (8):
        run<8, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<8>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (9):
        run<9, Kernel>(dc,
                       rhoc,
                       dm,
                       seed_dc,
                       pPBin,
                       std::move(wrapped),
                       std::make_tuple(pData, pResults),
                       n_points,
                       kernel,
                       tag_to_metric.template operator()<9>(metric),
                       queue,
                       block_size);
        return;
      [[unlikely]] case (10):
        run<10, Kernel>(dc,
                        rhoc,
                        dm,
                        seed_dc,
                        pPBin,
                        std::move(wrapped),
                        std::make_tuple(pData, pResults),
                        n_points,
                        kernel,
                        tag_to_metric.template operator()<10>(metric),
                        queue,
                        block_size);
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

}  // namespace ALPAKA_BACKEND
