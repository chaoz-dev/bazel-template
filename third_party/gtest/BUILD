cc_library(
    name = "gtest",
    testonly = True,
    visibility = ["//visibility:public"],
    deps = ["@com_github_google_googletest//:gtest_main"],
)

cc_test(
    name = "gtest_test",
    srcs = ["gtest_test.cc"],
    deps = [":gtest"],
)
