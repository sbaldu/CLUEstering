
#include <vector>

#include "../DataFormats/alpaka/Domains.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdint.h>

PYBIND11_MODULE(CLUE_Domains, m) {
  m.doc() = "Binding of the domain type used in CLUE.";

  pybind11::class_<domain_t>(m, "domain_t")
	.def(pybind11::init<>())
	.def(pybind11::init<float, float>());
}
