#include "third_party/cuda/test_kernel.cuh"

namespace third_party::cuda
{

__global__ void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

} // namespace third_party::cuda