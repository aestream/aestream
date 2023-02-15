#pragma once

#include <vector>

template <typename scalar_t>
void index_increment_cuda(scalar_t *array, std::vector<int> offsets,
                          int *event_device_pointer);
template <typename scalar_t> 
scalar_t *alloc_memory_cuda(size_t buffer_size);

template <typename scalar_t>
void free_memory_cuda(scalar_t *cuda_device_pointer);