
#include "defines.hpp"
#include <cstdint>
#include <memory>

namespace clue {
  template <uint8_t N>
  class Clusterer;
}

namespace serial {
  std::unique_ptr<clue::Clusterer<2>, void(*)(clue::Clusterer<2>*)> foo(float, float, float);
}

namespace cuda {
  std::unique_ptr<clue::Clusterer<2>, void(*)(clue::Clusterer<2>*)> foo(float, float, float);
}
