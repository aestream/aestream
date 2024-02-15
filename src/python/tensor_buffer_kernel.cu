#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

template <typename scalar_t>
__global__ void cuda_increment_kernel(scalar_t *__restrict__ array,
                                      int *__restrict__ offsets,
                                      const size_t size) {
  int index = threadIdx.x; // Pixel offset
  if (index < size) {
    float val = 1;
    atomicAdd((array + offsets[index]), val);
  }
}

void index_increment_cuda(float *array, int *offset_pointer, size_t indices,
                          int *event_device_pointer) {
  const size_t buffer_size = indices * sizeof(int);

  cudaMemcpyAsync(event_device_pointer, offset_pointer, buffer_size,
                  cudaMemcpyHostToDevice, cudaStreamPerThread);
  cuda_increment_kernel<float>
      <<<1, indices>>>(array, event_device_pointer, indices);
}

void *alloc_memory_cuda(size_t buffer_size, size_t bytes) {
  cudaError_t err;
  void *cuda_device_pointer;
  const size_t size = buffer_size * bytes;
  err = cudaMalloc(&cuda_device_pointer, size);
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "Error when allocating memory on GPU: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
  err = cudaMemsetAsync(cuda_device_pointer, 0, size, 0);
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "Error when resetting memory on GPU: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
  cudaDeviceSynchronize();
  return cuda_device_pointer;
}

void free_memory_cuda(void *cuda_device_pointer) {
  cudaError_t err = cudaFree(cuda_device_pointer);
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "Error when freeing memory on GPU: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}