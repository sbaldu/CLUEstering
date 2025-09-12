
#include "header.hpp"
#include "CLUEstering/CLUEstering.hpp"
#include "defines.hpp"

namespace backend {
  clue::Clusterer<2> foo(float dc, float rhoc, float dm) {
    auto c = clue::Clusterer<2>(dc, rhoc, dm);
    return c;
  }
}  // namespace backend
