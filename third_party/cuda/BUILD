load("@//rules:nvcc.bzl", "nvcc_test")

filegroup(
    name = "nvcc",
    srcs = ["bin/nvcc"],
    data = [":toolchain"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "toolchain",
    srcs = [
        "nvvm/bin/cicc",
    ] + glob([
        "nvvm/libdevice/libdevice.*.bc",
    ]) + glob([
        "bin/*",
    ]),
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = glob([
        "lib64/**/lib*.so",
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
    srcs = ["cuda_test.cc"],
    deps = ["//third_party/gtest"],
)
