load("//third_party/cuda:cuda.bzl", "cuda")
load("//third_party/glog:glog.bzl", "glog")
load("//third_party/gtest:gtest.bzl", "gtest")
load("//third_party/torch:torch.bzl", "torch")

def third_party():
    """Load third-party dependencies."""
    cuda()
    glog()
    gtest()
    torch()
