
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void cuda_increment_kernel(scalar_t *__restrict__ array, uint32_t * offsets, size_t size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
      *(array + offsets[index]) += 1;
    }
}

void index_increment_cuda(torch::Tensor array, std::vector<uint32_t> offsets, uint32_t* event_device_pointer) {
  const int indices = offsets.size();
  const size_t size = indices * sizeof(uint32_t);
  const int batch_size = 64;
  const int threads = indices / batch_size;
  const dim3 blocks((indices + threads - 1) / threads, batch_size);

  uint32_t* event_vector_pointer = &offsets[0];
  cudaMemcpy(event_device_pointer, event_vector_pointer, size, cudaMemcpyHostToDevice);

  AT_DISPATCH_INTEGRAL_TYPES(array.scalar_type(), "cuda_increment", ([&] {
    cuda_increment_kernel<scalar_t><<<blocks, threads>>>(
      array.data<scalar_t>(), event_device_pointer, indices);
  }));
}


uint32_t* alloc_memory_cuda(size_t buffer_size) {
  uint32_t *cuda_device_pointer;
  const size_t size = buffer_size * sizeof(uint32_t);
  cudaMalloc(&cuda_device_pointer, size);
  return cuda_device_pointer;
}
void free_memory_cuda(uint32_t* cuda_device_pointer) {
  cudaFree(cuda_device_pointer);
}