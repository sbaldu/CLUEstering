
#ifndef scaler_h
#define scaler_h

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <cstddef>
#include <cstdint>

#include "../AlpakaCore/alpakaWorkDiv.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "../DataFormats/alpaka/AlpakaVecArray.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct KernelFitTransform {
    template <typename TAcc, typename T>
    void operator()(const TAcc& acc, T* data, T mean, T std, size_t n_points) const {
      cms::alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
        data[i] -= mean;
        data[i] /= std;
      });
    }
  };

  template <typename T>
  void rescale(std::vector<T>& data, T mean, T std, size_t block_size = 256, size_t device_id = 0) {
    auto const dev_acc = alpaka::getDevByIdx<Acc1D>(device_id);
    Queue queue_(dev_acc);

    auto dev_buffer = cms::alpakatools::make_device_buffer<T[]>(queue_, data.size());
    alpaka::memcpy(queue_, dev_buffer, data.data(), data.size());
    const Idx grid_size = cms::alpakatools::divide_up_by(data.size(), block_size);
    auto work_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        work_div, KernelFitTransform(), data.data(), mean, std, data.size()));
    alpaka::memcpy(queue_, data.data(), dev_buffer, data.size());
    alpaka::wait(queue_);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
