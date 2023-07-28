cc_library(
    name = "glog",
    visibility = ["//visibility:public"],
    deps = [
        "@gflags",
        "@glog",
    ],
)

cc_test(
    name = "glog_test",
    srcs = ["glog_test.cc"],
    deps = [
        ":glog",
        "//third_party/gtest",
    ],
)
