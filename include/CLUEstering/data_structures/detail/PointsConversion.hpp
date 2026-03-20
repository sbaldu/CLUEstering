

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>

namespace clue {

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim, THostInput>& h_points,
                         const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
    auto& host_view = h_points.view();
    const auto& device_view = d_points.view();

    alpaka::memcpy(
        queue,
        make_host_view(host_view.m_cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), device_view.m_cluster_index, h_points.size()));
    h_points.mark_clustered();
    alpaka::wait(queue);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TInput, TDev>& d_points) {
    PointsHost<Ndim, std::remove_cv_t<TInput>> h_points(queue, d_points.size());
    auto& host_view = h_points.view();
    const auto& device_view = d_points.view();

    alpaka::memcpy(
        queue,
        make_host_view(host_view.m_cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), device_view.m_cluster_index, h_points.size()));
    h_points.mark_clustered();
    alpaka::wait(queue);

    return h_points;
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TDeviceInput,
            concepts::device TDev,
            std::floating_point THostInput>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                           const PointsHost<Ndim, THostInput>& h_points) {
    const auto& host_view = h_points.view();
    auto& device_view = d_points.view();

    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), device_view.m_coords[Dim], h_points.size()),
          make_host_view(host_view.m_coords[Dim], h_points.size()));
    });
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), device_view.m_weight, h_points.size()),
                   make_host_view(host_view.m_weight, h_points.size()));
    alpaka::wait(queue);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim, TInput>& h_points) {
    PointsDevice<Ndim, std::remove_cv_t<TInput>, TDev> d_points(queue, h_points.size());
    const auto& host_view = h_points.view();
    auto& device_view = d_points.view();

    meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), device_view.coords[Dim], h_points.size()),
          make_host_view(host_view.coords[Dim], h_points.size()));
    });
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), device_view.weight, h_points.size()),
                   make_host_view(host_view.weight, h_points.size()));
    alpaka::wait(queue);

    return d_points;
  }

}  // namespace clue
