alias(
    name = "torch",
    actual = "@org_pytorch_libtorch//:_torch",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_omp",
    srcs = ["libomp.so.5"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_python",
    srcs = ["libpython3.10.so"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_torch",
    srcs = glob([
        "libtorch/lib/**/*.so",
        "libtorch/lib/**/*.so.*",
        "libtorch/include/**/*.h",
    ]),
    defines = [
        # Enable glog support.
        "C10_USE_GLOG",
    ],
    includes = [
        "libtorch/include/",
        "libtorch/include/torch/csrc/api/include/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@//third_party/cuda",
        "@//third_party/glog",
        "@omp//:_omp",
        "@python//:_python",
    ],
)
