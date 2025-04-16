
#pragma once

#include "../AlpakaCore/alpakaMemory.hpp"

#include <algorithm>
#include <array>
#include <span>

namespace clue {

  template <uint8_t Ndim>
  class DefaultParameter {
	private:
	  std::array<float, Ndim> m_parameters;
	  clue::device_buffer<float[Ndim]> m_dparameters;
	
	public:
	  Parameter(float m) {
		std::fill(m_parameters.begin(), m_parameters.end(), m);
	  }
	  Parameter(const std::array<float, Ndim>& parameters) : m_parameters{parameters} {}
	  Parameter(std::array<float, Ndim>&& parameters) : m_parameters{std::move(parameters)} {}
	  template <typename... TArgs>
	  Parameter(TArgs... args) {
		static_assert(sizeof...(args) == Ndim);
		m_parameters = {static_cast<float>(args)...};
	  }

	  ALPAKA_FN_ACC auto operator()(float x, float y, float z) const {
		/* return m_parameters; */
	  }
  };

}
