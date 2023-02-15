#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void cuda_increment_kernel(scalar_t *__restrict__ array, int *__restrict__ offsets, const size_t size) {
    int index = threadIdx.x; // Pixel offset
    if (index < size) {
      array[offsets[index]]++;
    }
}

void index_increment_cuda(float *array, std::vector<int> offsets, int* event_device_pointer) {
  const size_t indices = offsets.size();
  const size_t buffer_size = indices * sizeof(int);

  int* event_vector_pointer = &offsets[0];
  cudaMemcpyAsync(event_device_pointer, event_vector_pointer, buffer_size, cudaMemcpyHostToDevice, 0);

  cuda_increment_kernel<float><<<1, indices>>>(array, event_device_pointer, indices);
}

float* alloc_memory_cuda_float(size_t buffer_size) {
  float *cuda_device_pointer;
  const size_t size = buffer_size * sizeof(float);
  cudaMalloc(&cuda_device_pointer, size);
  return cuda_device_pointer;
}

int* alloc_memory_cuda_int(size_t buffer_size) {
  int *cuda_device_pointer;
  const size_t size = buffer_size * sizeof(int);
  cudaMalloc(&cuda_device_pointer, size);
  return cuda_device_pointer;
}

void free_memory_cuda(float* cuda_device_pointer) {
  cudaFree(cuda_device_pointer);
}