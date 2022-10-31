#pragma once

namespace third_party::cuda
{

__global__ void add(int n, float* x, float* y);

} // namespace third_party::cuda