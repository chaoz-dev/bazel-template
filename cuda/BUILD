load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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
