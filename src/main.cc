#include "cuda/device_info.hh"

#include <iostream>

int main(int argc, char* argv[])
{
    cuda::print_devices();

    return 0;
}
