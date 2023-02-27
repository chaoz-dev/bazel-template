load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
load("//rules:nvcc.bzl", "nvcc_library", "nvcc_binary")

cc_library(
    name = "macros",
    hdrs = ["macros.hh"],
    visibility = ["//visibility:public"],
    deps = ["@cuda"],
)

cc_binary(
    name = "device_info",
    srcs = ["device_info.cc"],
    deps = [":macros"],
)

# Minimal example to show how to use nvcc_library and cc_binary to run cuda kernel
nvcc_library(
    name="minimal_example",
    srcs= ["minimal_example.cu.cc"],
    hdrs= ["minimal_example.hh"],
    )

cc_binary(
    name = "min_example",
    srcs = ["min_example.cc"],
    deps = [":minimal_example"],
)

# Reproducing the above example using nvcc_binary library

nvcc_binary(
    name="minimal_example_binary",
    srcs=["minimal_example.cu.cc"],
)








