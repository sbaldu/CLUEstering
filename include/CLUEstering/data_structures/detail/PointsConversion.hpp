

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Converter.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>
#include <concepts>
#include <cstddef>

namespace clue {

  template <std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  struct Converter<PointsHost<Ndim, THostInput>, PointsDevice<Ndim, TDeviceInput, TDev>> {
    template <concepts::queue TQueue, std::floating_point TDeviceInput, concepts::device TDev>
    static void to_device(TQueue& queue,
                          PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                          const PointsHost<Ndim, THostInput>& h_points) {
      meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
        alpaka::memcpy(
            queue,
            make_device_view(alpaka::getDev(queue), d_points.m_view.coords[Dim], h_points.size()),
            make_host_view(h_points.m_view.coords[Dim], h_points.size()));
      });
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
          make_host_view(h_points.m_view.weight, h_points.size()));
    }

    template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
    static auto to_device(TQueue& queue, const PointsHost<Ndim, THostInput>& h_points) {
      PointsDevice<Ndim, std::remove_cv_t<THostInput>, TDev> d_points(queue, h_points.size());

      meta::apply<Ndim>([&]<std::size_t Dim>() -> void {
        alpaka::memcpy(
            queue,
            make_device_view(alpaka::getDev(queue), d_points.m_view.coords[Dim], h_points.size()),
            make_host_view(h_points.m_view.coords[Dim], h_points.size()));
      });
      alpaka::memcpy(
          queue,
          make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
          make_host_view(h_points.m_view.weight, h_points.size()));

      return d_points;
    }

    template <concepts::queue TQueue, std::floating_point THostInput>
    static void to_host(TQueue& queue,
                        PointsHost<Ndim, THostInput>& h_points,
                        const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
      alpaka::memcpy(
          queue,
          make_host_view(h_points.m_view.cluster_index, h_points.size()),
          make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
      h_points.mark_clustered();
    }

    template <concepts::queue TQueue>
    static auto to_host(TQueue& queue, const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
      PointsHost<Ndim, std::remove_cv_t<TDeviceInput>> h_points(queue, d_points.size());

      alpaka::memcpy(
          queue,
          make_host_view(h_points.m_view.cluster_index, h_points.size()),
          make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
      h_points.mark_clustered();

      return h_points;
    }
  };

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point THostInput,
            std::floating_point TDeviceInput,
            concepts::device TDev>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim, THostInput>& h_points,
                         const PointsDevice<Ndim, TDeviceInput, TDev>& d_points) {
    return Converter<PointsHost<Ndim, THostInput>, PointsDevice<Ndim, TDeviceInput, TDev>>::to_host(
        queue, h_points, d_points);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToHost(TQueue& queue, const PointsDevice<Ndim, TInput, TDev>& d_points) {
    return Converter<PointsHost<Ndim, THostInput>, PointsDevice<Ndim, TDeviceInput, TDev>>::to_host(
        queue, d_points);
  }

  template <concepts::queue TQueue,
            std::size_t Ndim,
            std::floating_point TDeviceInput,
            concepts::device TDev,
            std::floating_point THostInput>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<Ndim, TDeviceInput, TDev>& d_points,
                           const PointsHost<Ndim, THostInput>& h_points) {
    return Converter<PointsHost<Ndim, THostInput>,
                     PointsDevice<Ndim, TDeviceInput, TDev>>::to_device(queue, d_points, h_points);
  }

  template <concepts::queue TQueue, std::size_t Ndim, std::floating_point TInput, concepts::device TDev>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim, TInput>& h_points) {
    return Converter<PointsHost<Ndim, THostInput>,
                     PointsDevice<Ndim, TDeviceInput, TDev>>::to_device(queue, h_points);
  }

}  // namespace clue
