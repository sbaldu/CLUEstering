
#include <cpp11/cpp11.hpp>

[[cpp11::register]]
template <std::floating_point TInput, template <typename T> typename Kernel>
  requires clue::concepts::convolutional_kernel<Kernel<TInput>>
void mainRun(TInput dc,
             TInput rhoc,
             TInput dm,
             TInput seed_dc,
             std::vector<uint8_t> wrapped,
             nb::ndarray<TInput, nb::numpy> data,
             nb::ndarray<int, nb::numpy> results,
             const Kernel<TInput>& kernel,
             int Ndim,
             std::optional<nb::ndarray<uint32_t, nb::numpy>> batch_sample_sizes,
             int32_t n_points,
             std::size_t device_id,
             const clue::bindings::MetricDescriptor<TInput>& metric_desc) {
  auto* pData = data.data();
  auto* pResults = results.data();
  const auto batch_sample_sizes_span = as_span(batch_sample_sizes);

  auto queue = clue::get_queue(device_id);
  const auto supported = dispatch_ndim(Ndim, [&]<std::size_t N>() {
    clue::PointsHost<N, TInput> h_points(queue, n_points, pData, pResults);
    run<TInput, N, Kernel<TInput>>(dc,
                                   rhoc,
                                   dm,
                                   seed_dc,
                                   std::move(wrapped),
                                   h_points,
                                   batch_sample_sizes_span,
                                   kernel,
                                   metric_desc,
                                   queue);
  });
  if (!supported) [[unlikely]] {
    std::cout << "The Python interface only supports up to 4 dimensions\n";
  }
}
