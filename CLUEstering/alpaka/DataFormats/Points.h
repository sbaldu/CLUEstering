#ifndef points_h
#define points_h

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include "alpaka/PointsAlpaka.h"
#include "alpaka/AlpakaVecArray.h"

using cms::alpakatools::VecArray;

template <uint8_t Ndim>
struct Points {
  Points() = default;
  Points(const std::vector<VecArray<float, Ndim>>& coords, const std::vector<float>& weight)
      : m_coords{coords}, m_weight{weight}, n{weight.size()} {}
  Points(const std::vector<std::vector<float>>& coords, const std::vector<float>& weight)
      : m_weight{weight}, n{weight.size()} {
    for (const auto& x : coords) {
      VecArray<float, Ndim> temp_vecarray;
      for (auto value : x) {
        temp_vecarray.push_back_unsafe(value);
      }
      m_coords.push_back(temp_vecarray);
    }

	std::vector<float> x(n);
	std::vector<float> y(n);
	std::vector<float> z(n);
	for (size_t j{}; j != n; ++j) {
	  x.push_back(m_coords[j][0]);
	  y.push_back(m_coords[j][1]);
	  z.push_back(m_coords[j][2]);
	}
	m_temp_coords.push_back(x);
	m_temp_coords.push_back(y);
	m_temp_coords.push_back(z);

    m_rho.resize(n);
    m_delta.resize(n);
    m_nearestHigher.resize(n);
    m_clusterIndex.resize(n);
    m_isSeed.resize(n);
  }

  std::vector<std::vector<float>> m_temp_coords;
  std::vector<VecArray<float, Ndim>> m_coords;
  std::vector<float> m_weight;
  std::vector<float> m_rho;
  std::vector<float> m_delta;
  std::vector<int> m_nearestHigher;
  std::vector<int> m_clusterIndex;
  std::vector<int> m_isSeed;

  size_t n;
};

#endif
