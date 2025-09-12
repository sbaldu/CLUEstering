
#include "defines.hpp"
#include <cstdint>

namespace clue {
  template <uint8_t N>
  class Clusterer;
}

namespace serial {
  clue::Clusterer<2> foo(float, float, float);
}

namespace cuda {
  clue::Clusterer<2> foo(float, float, float);
}
