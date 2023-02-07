#include "tensor_buffer_kernel.h"

template <typename scalar_t>
__global__ void cuda_increment_kernel(scalar_t *__restrict__ array, int *__restrict__ offsets, const size_t size) {
    int index = threadIdx.x; // Pixel offset
    if (index < size) {
      array[offsets[index]]++;
    }
}

template <typename scalar_t>
void index_increment_cuda(scalar_t *array, std::vector<int> offsets, int* event_device_pointer) {
  const size_t indices = offsets.size();
  const size_t buffer_size = indices * sizeof(int);

  int* event_vector_pointer = &offsets[0];
  cudaMemcpyAsync(event_device_pointer, event_vector_pointer, buffer_size, cudaMemcpyHostToDevice, 0);

  cuda_increment_kernel<float><<<1, indices>>>(array, event_device_pointer, indices);
}

template <typename scalar_t>
scalar_t* alloc_memory_cuda(size_t buffer_size) {
  scalar_t *cuda_device_pointer;
  const size_t size = buffer_size * sizeof(scalar_t);
  cudaMalloc(&cuda_device_pointer, size);
  return cuda_device_pointer;
}

template <typename scalar_t>
void free_memory_cuda(scalar_t* cuda_device_pointer) {
  cudaFree(cuda_device_pointer);
}