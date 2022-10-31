#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace cuda
{

#define CUDA_ASSERT(err_n) ::cuda::cuda_assert(err_n, true, __FILE__, __LINE__);
inline void cuda_assert(cudaError_t err_n, bool terminate, const char* filename, int lineno)
{
    if (err_n == cudaSuccess)
    {
        return;
    }

    std::cerr << filename << ": " << lineno << std::endl
              << cudaGetErrorName(err_n) << ": " << std::endl
              << cudaGetErrorString(err_n) << std::endl;

    if (terminate)
    {
        std::exit(EXIT_FAILURE);
    }
}

} // namespace cuda