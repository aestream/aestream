
#include <torch/extension.h>

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

void index_increment_cuda(torch::Tensor array, std::vector<int> offsets, int* event_device_pointer) {
  const size_t indices = offsets.size();
  const size_t buffer_size = indices * sizeof(int);

  int* event_vector_pointer = &offsets[0];
  cudaMemcpy(event_device_pointer, event_vector_pointer, buffer_size, cudaMemcpyHostToDevice);

  AT_DISPATCH_INTEGRAL_TYPES(array.scalar_type(), "cuda_increment", ([&] {
    cuda_increment_kernel<scalar_t><<<1, indices>>>(
      array.data_ptr<scalar_t>(), event_device_pointer, indices);
  }));
}


int* alloc_memory_cuda(size_t buffer_size) {
  int *cuda_device_pointer;
  const size_t size = buffer_size * sizeof(int);
  cudaMalloc(&cuda_device_pointer, size);
  return cuda_device_pointer;
}
void free_memory_cuda(int* cuda_device_pointer) {
  cudaFree(cuda_device_pointer);
}