#include "convert.hpp"

// torch::Tensor
// convert_polarity_events(std::vector<AEDAT::PolarityEvent> &polarity_events,
//                         const std::vector<int64_t> &tensor_size) {
//   const size_t size = polarity_events.size();
//   std::vector<int64_t> indices(3 * size);
//   std::vector<int8_t> values;
//   const auto max_duration =
//       tensor_size.empty()
//           ? polarity_events.back().timestamp - polarity_events[0].timestamp
//           : tensor_size[0];

//   for (size_t idx = 0; idx < size; idx++) {
//     auto event = polarity_events[idx];
//     auto event_time = event.timestamp - polarity_events[0].timestamp;
//     //  Break if event is after max_duration
//     if (event_time >= max_duration) {
//       break;
//     }

//     indices[idx] = event_time;
//     indices[size + idx] = event.x;
//     indices[2 * size + idx] = event.y;
//     values.push_back(event.polarity ? 1 : -1);
//   }

//   auto index_options = torch::TensorOptions().dtype(torch::kInt64);
//   torch::Tensor ind = torch::from_blob(
//       indices.data(), {3, static_cast<uint32_t>(size)}, index_options);

//   auto value_options = torch::TensorOptions().dtype(torch::kInt8);
//   torch::Tensor val = torch::from_blob(
//       values.data(), {static_cast<uint32_t>(size)}, value_options);

//   auto events =
//       tensor_size.empty()
//           ? torch::sparse_coo_tensor(ind, val)
//           : torch::sparse_coo_tensor(ind, val,
//           torch::IntArrayRef(tensor_size));

//   return events.clone();
// }

// std::vector<torch::Tensor>
// convert_polarity(std::vector<AEDAT::PolarityEvent> &polarity_events,
//                  const int64_t window_size,
//                  const int64_t window_step,
//                  const std::vector<double> &scale,
//                  const std::vector<int64_t> &image_dimensions) {
//   std::vector<torch::Tensor> event_tensors;
//   size_t start = 0;
//   size_t idx = 0;
//   size_t next_idx = 0;
//   bool next_idx_found = false;
//   auto last_event = polarity_events.back();
//   while (start <  last_event.timestamp - window_size) {
//     auto event = polarity_events[idx];
//     size_t start_time = event.timestamp;
//     std::vector<int64_t> indices;
//     std::vector<int8_t> values;

//     while (event.timestamp < start + window_size) {
//         indices.push_back(static_cast<int64_t>((event.timestamp -
//         start_time)/scale[0]));
//         indices.push_back(static_cast<int64_t>(event.x/scale[1]));
//         indices.push_back(static_cast<int64_t>(event.y/scale[2]));
//         values.push_back(event.polarity ? 1 : -1);
//         if (!next_idx_found && (event.timestamp >= start + window_step)) {
//             next_idx = idx;
//             next_idx_found = true;
//         }
//         idx += 1;
//         event = polarity_events[idx];
//     }

//     // create sparse tensor
//     auto index_options = torch::TensorOptions().dtype(torch::kInt64);
//     torch::Tensor ind = torch::from_blob(
//       indices.data(), {static_cast<uint32_t>(indices.size() / 3), 3},
//       index_options).permute({1,0});

//     auto value_options = torch::TensorOptions().dtype(torch::kInt8);
//     torch::Tensor val = torch::from_blob(
//       values.data(), {static_cast<uint32_t>(indices.size()/3)},
//       value_options);
//     auto events = torch::sparse_coo_tensor(ind, val, {window_size,
//     image_dimensions[0], image_dimensions[1]});

//     event_tensors.push_back(events.clone());

//     idx = next_idx;
//     start += window_step;
//     next_idx_found = false;
//   }

//   return event_tensors;
// }