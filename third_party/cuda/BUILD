load("@//rules:nvcc.bzl", "nvcc_test")

filegroup(
    name = "toolchain",
    srcs = [
        "nvvm/bin/cicc",
    ] + glob([
        "nvvm/libdevice/libdevice.*.bc",
    ]) + glob([
        "bin/*",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nvcc",
    srcs = ["bin/nvcc"],
    data = [":toolchain"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = glob(["lib64/**/lib*.so"]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath=lib64/"],
    visibility = ["//visibility:public"],
)

nvcc_test(
    name = "cuda_test",
    srcs = [
        "cuda_test.cc",
        "test_kernel.cu",
    ],
    hdrs = ["test_kernel.cuh"],
    deps = ["@gtest//:gtest_main"],
)
