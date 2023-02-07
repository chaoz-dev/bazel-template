cc_library(
    name = "glog",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
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
