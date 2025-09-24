
#pragma once

#include <algorithm>
#include <ranges>
#include <span>
#include <vector>

namespace clue {

  inline bool validate_results(std::span<const int> cluster_ids, std::span<const int> truth) {
    auto result_clusters_sizes = compute_clusters_size(cluster_ids);
    auto truth_clusters_sizes = compute_clusters_size(truth);
    std::ranges::sort(result_clusters_sizes);
    std::ranges::sort(truth_clusters_sizes);

    bool compare_nclusters = compute_nclusters(cluster_ids) == compute_nclusters(truth);
    bool compare_clusters_size = std::ranges::equal(result_clusters_sizes, truth_clusters_sizes);

    return compare_nclusters && compare_clusters_size;
  }

}  // namespace clue::internal
