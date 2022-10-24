load("//rules:nvcc.bzl", "nvcc_library")

package(default_visibility = ["//visibility:public"])

nvcc_library(
    name = "add",
    srcs = ["add.cu.cc"],
    hdrs = ["add.cu.hh"],
    deps = [
        "//cuda:macros",
    ],
)
