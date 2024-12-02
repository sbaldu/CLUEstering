#ifndef run_h
#define run_h

#include <vector>

#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim, typename Kernel>
  void run(float dc,
           float rhoc,
           float dm,
           int pPBin,
           std::tuple<float*, int*>&& pData,
           const PointShape<Ndim>& shape,
           const Kernel& kernel,
           Queue queue_,
           size_t block_size) {
    CLUEAlgoAlpaka<Acc1D, Ndim> algo(dc, rhoc, dm, pPBin, queue_);

    // Create the host and device points
    PointsSoA<Ndim> h_points(std::get<0>(pData), std::get<1>(pData), shape);
    PointsAlpaka<Ndim> d_points(queue_, shape.nPoints);

    algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
