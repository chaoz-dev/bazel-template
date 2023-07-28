cc_library(
    name = "gtest",
    testonly = True,
    visibility = ["//visibility:public"],
    deps = ["@gtest//:gtest_main"],
)

cc_test(
    name = "gtest_test",
    srcs = ["gtest_test.cc"],
    deps = [":gtest"],
)
