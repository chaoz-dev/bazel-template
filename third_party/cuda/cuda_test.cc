#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

TEST(Cuda, DeviceCheck)
{
    int num_devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
}

TEST(Cuda, KernelCheck)
{
    int num_devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
    ASSERT_GT(num_devices, 0);

    constexpr int N = 1 << 10;
    float *x, *y;

    ASSERT_EQ(cudaMallocManaged(&x, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&y, N * sizeof(float)), cudaSuccess);

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float max_error = 0;
    for (int i = 0; i < N; ++i)
    {
        max_error = std::max(max_error, std::fabs(y[i] - 3.0f));
    }
    EXPECT_EQ(max_error, 0.0);

    ASSERT_EQ(cudaFree(x), cudaSuccess);
    ASSERT_EQ(cudaFree(y), cudaSuccess);
}