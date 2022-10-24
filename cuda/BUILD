cc_library(
    name = "device_info",
    srcs = ["device_info.cc"],
    hdrs = ["device_info.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":macros",
        "@cuda",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.hh"],
    visibility = ["//cuda:__subpackages__"],
    deps = ["@cuda"],
)
