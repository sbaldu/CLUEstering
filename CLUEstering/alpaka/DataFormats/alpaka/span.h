
#pragma once

#include <cstdint>
#include <type_traits>

namespace clue {

  template <typename T>
  class span {
  private:
    T* m_data;
    uint32_t m_size;

  public:
    typedef T element_type;
    typedef std::remove_cv_t<T> value_type;
    typedef uint32_t size_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef T& reference;
    typedef const T& const_reference;

    span() = delete;
    template <typename E, typename = std::enable_if_t<std::is_convertible<E, T>::value>>
    ALPAKA_FN_HOST_ACC span(E* data, uint32_t size) : m_data{data}, m_size{size} {}

    ALPAKA_FN_HOST_ACC inline pointer* data() { return m_data; }
    ALPAKA_FN_HOST_ACC inline const_pointer* data() const { return m_data; }

    ALPAKA_FN_HOST_ACC inline size_type size() const { return m_size; }

    ALPAKA_FN_HOST_ACC inline reference operator[](uint32_t i) { return m_data[i]; }
    ALPAKA_FN_HOST_ACC inline const_reference operator[](uint32_t i) const {
      return m_data[i];
    }

    ALPAKA_FN_HOST_ACC inline iterator begin() { return m_data; }
    ALPAKA_FN_HOST_ACC inline const_iterator begin() const { return m_data; }

    ALPAKA_FN_HOST_ACC inline iterator end() { return m_data + m_size; }
    ALPAKA_FN_HOST_ACC inline const_iterator end() const { return m_data + m_size; }
  };

};  // namespace clue
