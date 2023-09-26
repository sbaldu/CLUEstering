#ifndef domains_h
#define domains_h

#include <array>
#include <cstdint>
#include <limits>

#include "../../AlpakaCore/alpakaConfig.h"
#include "../../AlpakaCore/alpakaMemory.h"
#include "AlpakaVecArray.h"

using cms::alpakatools::VecArray;

class domain_t {
private:
  float m_min;
  float m_max;

public:
  domain_t() : m_min{-std::numeric_limits<float>::max()}, m_max{std::numeric_limits<float>::max()} {}
  domain_t(float min, float max) : m_min{min}, m_max{max} {}
  float min() const { return m_min; }
  float max() const { return m_max; }
};

template <uint8_t Ndim>
class domain_ranges {
private:
  VecArray<domain_t, Ndim> domains_;

public:
  domain_ranges() = delete;
  domain_ranges(const std::array<domain_t, Ndim>& domains) {
    for (int dim{}; dim < Ndim; ++dim) {
      domain_t temp(domains[dim].min(), domains[dim].max());
      domains_.push_back_unsafe(temp);
    }
  }
  domain_ranges(const std::vector<domain_t>& domains) {
    for (size_t dim{}; dim < domains.size(); ++dim) {
      domain_t temp(domains[dim].min(), domains[dim].max());
      domains_.push_back_unsafe(temp);
    }
  }

  // Getters
  VecArray<domain_t, Ndim> data() { return domains_; }
  VecArray<domain_t, Ndim>* get() { return domains_[0].begin(); }
};

#endif
