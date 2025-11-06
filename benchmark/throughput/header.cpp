
#include "header.hpp"
#include "CLUEstering/CLUEstering.hpp"
#include "defines.hpp"

namespace backend {
  ::std::unique_ptr<clue::Clusterer<2>, void (*)(clue::Clusterer<2>*)> foo(float dc,
                                                                         float rhoc,
                                                                         float dm) {
	return ::std::unique_ptr<clue::Clusterer<2>, void (*)(clue::Clusterer<2>*)>(
        new clue::Clusterer<2>(dc, rhoc, dm), [](clue::Clusterer<2>* ptr) { delete ptr; });
  }
}  // namespace backend
