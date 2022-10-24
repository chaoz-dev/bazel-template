#include "cuda/kernels/add.cu.hh"

#include "cuda/macros.hh"

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

namespace cuda
{
namespace
{

__global__ void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

} // namespace

void add()
{
    constexpr int N = 1 << 20;
    float *x, *y;

    CUDA_ASSERT(cudaMallocManaged(&x, N * sizeof(float)));
    CUDA_ASSERT(cudaMallocManaged(&y, N * sizeof(float)));

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    CUDA_ASSERT(cudaDeviceSynchronize());

    float max_error = 0;
    for (int i = 0; i < N; ++i)
    {
        max_error = std::max(max_error, std::fabs(y[i] - 3.0f));
    }
    std::cout << "Max Error: " << max_error << std::endl;

    CUDA_ASSERT(cudaFree(x));
    CUDA_ASSERT(cudaFree(y));
}

} // namespace cuda