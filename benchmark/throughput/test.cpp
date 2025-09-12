
#include "header.hpp"
#include <iostream>

int main() {
  auto c1 = serial::foo(1.f, 1.f, 1.f);
  auto c2 = cuda::foo(1.f, 1.f, 1.f);
}
