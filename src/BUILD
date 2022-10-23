load("//rules:nvcc.bzl", "nvcc_library")

nvcc_library(
    name = "saxpy",
    srcs = ["saxpy.cu"],
    hdrs = ["saxpy.cuh"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":saxpy",
    ],
)
