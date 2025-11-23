
#pragma once

#include <type_traits>

namespace clue {

  struct neighborhood_policy {};

  struct round_neighborhood_policy : public neighborhood_policy {};
  struct square_neighborhood_policy : public neighborhood_policy {};

  inline constexpr auto round_neighborhood = round_neighborhood_policy{};
  inline constexpr auto square_neighborhood = square_neighborhood_policy{};

  template <typename Policy>
  concept neighbohood = std::is_base_of_v<neighborhood_policy, Policy>;

}  // namespace clue
