#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void print_kernel_id() {
    print_kernel<<<10, 10>>>();
    cudaDeviceSynchronize();
}

int main(){
    print_kernel_id();
    return 0;
}

