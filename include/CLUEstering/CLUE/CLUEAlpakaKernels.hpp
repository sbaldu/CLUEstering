
#pragma once

#include <alpaka/core/Common.hpp>
#include <chrono>
#include <cstdint>

#include "../AlpakaCore/alpakaWorkDiv.hpp"
#include "../DataFormats/alpaka/PointsAlpaka.hpp"
#include "../DataFormats/alpaka/TilesAlpaka.hpp"
#include "../DataFormats/alpaka/AlpakaVecArray.hpp"
#include "ConvolutionalKernel.hpp"

using clue::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  constexpr int32_t max_followers{100};
  constexpr int32_t reserve{1000000};

  template <uint8_t Ndim>
  ALPAKA_FN_ACC void getCoords(float* coords, PointsAlpakaView* d_points, uint32_t i) {
    for (auto dim = 0; dim < Ndim; ++dim) {
      coords[dim] = d_points->coords[i + dim * d_points->n];
    }
  }

  template <uint8_t Ndim, uint8_t N_, typename KernelType>
  ALPAKA_FN_HOST_ACC void for_recursion(
      const TAcc& acc,
      VecArray<uint32_t, Ndim>& base_vec,
      const VecArray<VecArray<uint32_t, 2>, Ndim>& search_box,
      TilesAlpakaView<Ndim>* tiles,
      PointsAlpakaView* dev_points,
      const KernelType& kernel,
      const float* coords_i,
      float* rho_i,
      float dc,
      uint32_t point_id) {
    if constexpr (N_ == 0) {
      auto binId = tiles->getGlobalBinByBin(acc, base_vec);
      // get the size of this bin
      auto binSize = (*tiles)[binId].size();

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        uint32_t j{(*tiles)[binId][binIter]};
        // query N_{dc_}(i)

        float coords_j[Ndim];
        getCoords<Ndim>(coords_j, dev_points, j);

        float dist_ij_sq{0.f};
        for (int dim{}; dim != Ndim; ++dim) {
          dist_ij_sq += (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (dist_ij_sq <= dc * dc) {
          *rho_i += kernel(acc, alpaka::math::sqrt(acc, dist_ij_sq), point_id, j) *
                    dev_points->weight[j];
        }

      }  // end of interate inside this bin
      return;
    } else {
      for (unsigned int i{search_box[search_box.capacity() - N_][0]};
           i <= search_box[search_box.capacity() - N_][1];
           ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion<TAcc, Ndim, N_ - 1>(acc,
                                          base_vec,
                                          search_box,
                                          tiles,
                                          dev_points,
                                          kernel,
                                          coords_i,
                                          rho_i,
                                          dc,
                                          point_id);
      }
    }
  }

  template <uint8_t Ndim, uint8_t N_>
  ALPAKA_FN_HOST_ACC void for_recursion_nearest_higher(
      const Acc1D& acc,
      VecArray<uint32_t, Ndim>& base_vec,
      const VecArray<VecArray<uint32_t, 2>, Ndim>& s_box,
      TilesAlpakaView<Ndim>* tiles,
      PointsAlpakaView* dev_points,
      const float* coords_i,
      float rho_i,
      float* delta_i,
      int* nh_i,
      float dm_sq,
      uint32_t point_id) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(acc, base_vec)};
      // get the size of this bin
      int binSize{(*tiles)[binId].size()};

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        unsigned int j{(*tiles)[binId][binIter]};
        // query N'_{dm}(i)
        float rho_j{dev_points->rho[j]};
        bool found_higher{(rho_j > rho_i)};
        // in the rare case where rho is the same, use detid
        found_higher =
            found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

        // Calculate the distance between the two points
        float coords_j[Ndim];
        getCoords<Ndim>(coords_j, dev_points, j);

        float dist_ij_sq{0.f};
        for (int dim{}; dim != Ndim; ++dim) {
          dist_ij_sq += (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (found_higher && dist_ij_sq <= dm_sq) {
          // find the nearest point within N'_{dm}(i)
          if (dist_ij_sq < *delta_i) {
            // update delta_i and nearestHigher_i
            *delta_i = dist_ij_sq;
            *nh_i = j;
          }
        }
      }  // end of interate inside this bin

      return;
    } else {
      for (unsigned int i{s_box[s_box.capacity() - N_][0]};
           i <= s_box[s_box.capacity() - N_][1];
           ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion_nearest_higher<TAcc, Ndim, N_ - 1>(acc,
                                                         base_vec,
                                                         s_box,
                                                         tiles,
                                                         dev_points,
                                                         coords_i,
                                                         rho_i,
                                                         delta_i,
                                                         nh_i,
                                                         dm_sq,
                                                         point_id);
      }
    }
  }


  struct KernelResetFollowers {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  VecArray<int, max_followers>* d_followers,
                                  uint32_t n_points) const;
  };

  struct KernelCalculateLocalDensity {
    template <uint8_t Ndim, typename KernelType>
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  TilesAlpakaView<Ndim>* dev_tiles,
                                  PointsAlpakaView* dev_points,
                                  const KernelType& kernel,
                                  float dc,
                                  uint32_t n_points) const;
  };

  struct KernelCalculateNearestHigher {
    template <uint8_t Ndim>
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  TilesAlpakaView<Ndim>* dev_tiles,
                                  PointsAlpakaView* dev_points,
                                  float dm,
                                  float,
                                  uint32_t n_points) const;
  };

  struct KernelFindClusters {
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int32_t, reserve>* seeds,
                                  VecArray<int32_t, max_followers>* followers,
                                  PointsAlpakaView* dev_points,
                                  float dm,
                                  float d_c,
                                  float rho_c,
                                  uint32_t n_points) const;
  };

  struct KernelAssignClusters {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  VecArray<int32_t, reserve>* seeds,
                                  VecArray<int, max_followers>* followers,
                                  PointsAlpakaView* dev_points) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
