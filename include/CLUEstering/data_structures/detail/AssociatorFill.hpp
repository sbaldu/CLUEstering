
#pragma once

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>
#include <span>

namespace clue::detail {

  struct KernelComputeAssociationSizes {
    template <typename TAcc>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const int32_t* associations,
                                  int32_t* bin_sizes,
                                  std::size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        if (associations[i] >= 0)
          alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1);
      }
    }
  };

  struct KernelFillAssociator {
    template <typename TAcc>
      requires(alpaka::Dim<TAcc>::value == 1)
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  int32_t* indexes,
                                  const int32_t* bin_buffer,
                                  int32_t* temp_offsets,
                                  size_t size) const {
      for (auto i : alpaka::uniformElements(acc, size)) {
        const auto binId = bin_buffer[i];
        if (binId >= 0) {
          auto prev = alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
          indexes[prev] = i;
        }
      }
    }
  };

  template <concepts::accelerator TAcc, concepts::queue TQueue>
    requires(alpaka::Dim<TAcc>::value == 1)
  ALPAKA_FN_HOST inline auto compute_key_counts(TQueue& queue,
                                                std::span<const int32_t> keys,
                                                int32_t nkeys) {
    auto keys_counts = make_device_buffer<key_type[]>(queue, nkeys);
    alpaka::memset(queue, keys_counts, 0);
    alpaka::exec<TAcc>(queue,
                       workdiv,
                       detail::KernelComputeAssociationSizes{},
                       keys.data(),
                       keys_counts.data(),
                       keys.size());
    return keys_counts;
  }

  template <concepts::accelerator TAcc, concepts::queue TQueue>
    requires(alpaka::Dim<TAcc>::value == 1)
  ALPAKA_FN_HOST inline void compute_key_offsets(TQueue& queue, ) {
    auto temp_offsets = make_device_buffer<key_type[]>(queue, m_extents.keys + 1);
    alpaka::memset(queue, temp_offsets, 0u, 1u);
    const auto blocksize_multiblockscan = 1024;
    auto gridsize_multiblockscan = divide_up_by(m_extents.keys, blocksize_multiblockscan);
    const auto workdiv_multiblockscan =
        make_workdiv<TAcc>(gridsize_multiblockscan, blocksize_multiblockscan);
    const auto dev = alpaka::getDev(queue);

    auto warp_size = alpaka::getPreferredWarpSize(dev);
    alpaka::exec<TAcc>(queue,
                       workdiv_multiblockscan,
                       multiBlockPrefixScan<key_type>{},
                       sizes_buffer.data(),
                       temp_offsets.data() + 1,
                       m_extents.keys,
                       gridsize_multiblockscan,
                       block_counter.data(),
                       warp_size);

    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), m_offsets.data(), m_extents.keys + 1),
                   temp_offsets);
  }

  template <concepts::accelerator TAcc, concepts::queue TQueue>
    requires(alpaka::Dim<TAcc>::value == 1)
  ALPAKA_FN_HOST inline void fill_associator() {}

}  // namespace clue::detail
