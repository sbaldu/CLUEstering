#ifndef Points_Alpaka_h
#define Points_Alpaka_h

#include <cstdint>
#include <memory>

#include "../../AlpakaCore/alpakaConfig.hpp"
#include "../../AlpakaCore/alpakaMemory.hpp"
#include "../Points.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  class PointsAlpakaView {
  public:
    float* coords;
    float* weight;
    float* rho;
    float* delta;
    int* nearest_higher;
    int* cluster_index;
    int* is_seed;
    int n;
  };

  template <uint8_t Ndim>
  class PointsAlpaka {
  public:
    explicit PointsAlpaka(Queue queue, int n_points)
        : input_buffer{clue::make_device_buffer<float[]>(queue, (Ndim + 3) * n_points)},
          result_buffer{clue::make_device_buffer<int[]>(queue, 3 * n_points)},
          view_dev{clue::make_device_buffer<PointsAlpakaView>(queue)} {
      auto view_host = clue::make_host_buffer<PointsAlpakaView>(queue);
      view_host->coords = input_buffer.data();
      view_host->weight = input_buffer.data() + Ndim * n_points;
      view_host->rho = input_buffer.data() + (Ndim + 1) * n_points;
      view_host->delta = input_buffer.data() + (Ndim + 2) * n_points;
      view_host->nearest_higher = result_buffer.data();
      view_host->cluster_index = result_buffer.data() + n_points;
      view_host->is_seed = result_buffer.data() + 2 * n_points;
      view_host->n = n_points;

      alpaka::memcpy(queue, view_dev, view_host);
    }
    explicit PointsAlpaka(Queue queue, float* input_data, int* result_data, int n_points)
        : input_buffer{clue::make_device_buffer<float[]>(queue, 2 * n_points)},
          result_buffer{clue::make_device_buffer<int[]>(queue, n_points)},
          view_dev{clue::make_device_buffer<PointsAlpakaView>(queue)} {
      auto h_view = clue::make_host_buffer<PointsAlpakaView>();
      h_view->coords = input_data;
      h_view->weight = input_data + Ndim * n_points;
      h_view->rho = input_buffer.data();
      h_view->delta = input_buffer.data() + n_points;
      h_view->nearest_higher = result_buffer.data();
      h_view->cluster_index = result_data;
      h_view->is_seed = result_data + n_points;
      h_view->n = n_points;
      alpaka::memcpy(queue, view_dev, h_view);
    }

    PointsAlpaka(const PointsAlpaka&) = delete;
    PointsAlpaka& operator=(const PointsAlpaka&) = delete;
    PointsAlpaka(PointsAlpaka&&) = default;
    PointsAlpaka& operator=(PointsAlpaka&&) = default;
    ~PointsAlpaka() = default;

    clue::device_buffer<Device, float[]> input_buffer;
    clue::device_buffer<Device, int[]> result_buffer;

    PointsAlpakaView* view() { return view_dev.data(); }

  private:
    clue::device_buffer<Device, PointsAlpakaView> view_dev;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

#endif
