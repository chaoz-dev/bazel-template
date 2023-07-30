#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

__global__ void print_kernel_id()
{
    printf("Hello world from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

void print_kernel_ids()
{
    print_kernel_id<<<3, 3>>>();

    cudaDeviceSynchronize();
}

// The `ifdef` below is not required by default. However, by wrapping the `main` function in an `ifdef NVCC_BINARY`, we
// can build this file as both an `nvcc_library` to a `cc_binary` and as a standalone `nvcc_binary`.

#ifdef NVCC_BINARY
int main()
{
    print_kernel_ids();

    return 0;
}
#endif
