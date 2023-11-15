#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

TEST(LibTorch, SanityCheck) {
    torch::manual_seed(0);
    const auto t = torch::rand({2, 3});
    EXPECT_TRUE(torch::allclose(t, t));
}
