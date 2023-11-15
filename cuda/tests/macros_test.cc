#include "cuda/macros.hh"

#include <gtest/gtest.h>

TEST(CudaAssert, Pass)
{
    ASSERT_NO_THROW(CUDA_ASSERT(cudaError::cudaSuccess));
}

TEST(CudaAssert, Fail)
{
    ASSERT_DEATH(CUDA_ASSERT(cudaError::cudaErrorInvalidValue), "");
}
