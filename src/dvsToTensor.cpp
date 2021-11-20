#include <chrono>
#include <torch/torch.h>

#include "aedat.hpp"
#include "convert.hpp"
#include "dvsToTensor.hpp"
#include "generator.hpp"

// dense tensor generator
Generator<torch::Tensor>
sparse_tensor_generator(Generator<AEDAT::PolarityEvent>& event_generator,
                       std::chrono::duration<double, std::micro> event_window) {

  std::vector<AEDAT::PolarityEvent> polarity_events;
  auto start = std::chrono::steady_clock::now();

  for (AEDAT::PolarityEvent event : event_generator) {
    polarity_events.push_back(event);

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    start = std::chrono::steady_clock::now();
    if (duration > event_window) {
      auto event_tensors = convert_polarity_events(polarity_events);
      co_yield event_tensors;

      std::vector<AEDAT::PolarityEvent>().swap(polarity_events);
    }
  }
}
