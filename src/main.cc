#include "cuda/device_info.hh"
#include "cuda/kernels/add.cu.hh"

#include <iostream>

int main(int argc, char* argv[])
{
    cuda::print_devices();
    cuda::add();

    return 0;
}
