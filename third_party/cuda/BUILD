alias(
    name = "cuda",
    actual = "@cuda//:_cuda",
    visibility = ["//visibility:public"],
)

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
    name = "_cuda",
    srcs = glob([
        "lib64/**/lib*.so",
    ]),
    hdrs = glob([
        "include/nv/**/*",
        "include/cuda/**/*",
        "include/cub/**/*",
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath=lib64/"],
    visibility = ["//visibility:public"],
)
