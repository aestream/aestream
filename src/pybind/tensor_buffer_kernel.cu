#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

template <typename scalar_t>
__global__ void cuda_increment_kernel(scalar_t *__restrict__ array, int *__restrict__ offsets, const size_t size) {
    int index = threadIdx.x; // Pixel offset
    if (index < size) {
      float val = 1;
      atomicAdd((array + offsets[index]), val);
    }
}

void index_increment_cuda(float *array, int *offset_pointer, size_t indices, int* event_device_pointer) {
  const size_t buffer_size = indices * sizeof(int);

  cudaMemcpyAsync(event_device_pointer, offset_pointer, buffer_size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  cuda_increment_kernel<float><<<1, indices>>>(array, event_device_pointer, indices);
}

void* alloc_memory_cuda(size_t buffer_size, size_t bytes) {
  void *cuda_device_pointer;
  const size_t size = buffer_size * bytes;
  cudaMalloc(&cuda_device_pointer, size);
  cudaMemset(&cuda_device_pointer, 0, size);
  return cuda_device_pointer;
}

void free_memory_cuda(void* cuda_device_pointer) {
  cudaFreeAsync(cuda_device_pointer, 0);
}