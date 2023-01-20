#include "dvs_to_tensor.hpp"

torch::Tensor
convert_polarity_events(std::vector<AER::Event> &polarity_events,
                        const torch::IntArrayRef &shape,
                        const torch::Device device) {
  const int64_t size = polarity_events.size();
  auto ind = torch::empty(
      {4, size}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  int64_t *indices = (int64_t *)ind.data_ptr();
  auto val = torch::ones(
      {size}, torch::TensorOptions().dtype(torch::kInt16).device(device));

  for (size_t idx = 0; idx < size; idx++) {
    auto event = polarity_events[idx];
    auto event_time = event.timestamp - polarity_events[0].timestamp;

    indices[idx] = event_time;
    indices[size + idx] = event.polarity;
    indices[2 * size + idx] = event.x;
    indices[3 * size + idx] = event.y;
  }

  auto sparse_options =
      torch::TensorOptions().dtype(torch::kInt16).device(device);

  if (shape.empty()) {
    return torch::sparse_coo_tensor(ind, val, sparse_options);
  } else {
    return torch::sparse_coo_tensor(
        ind, val, {indices[size - 1] + 1, 2, shape[0], shape[1]},
        sparse_options);
  }
}

Generator<torch::Tensor>
sparse_tensor_generator(Generator<AER::Event> &event_generator,
                        std::chrono::duration<double, std::micro> event_window,
                        const torch::IntArrayRef shape,
                        const torch::Device device) {

  std::vector<AER::Event> polarity_events;
  auto start = std::chrono::high_resolution_clock::now();

  for (AER::Event event : event_generator) {
    polarity_events.push_back(event);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration >= event_window) {
      start = std::chrono::high_resolution_clock::now();
      co_yield convert_polarity_events(polarity_events, shape, device);
      polarity_events.clear();
    }
  }
  // Send remaining events
  if (polarity_events.size() > 0) {
    co_yield convert_polarity_events(polarity_events, shape, device);
  }
}
