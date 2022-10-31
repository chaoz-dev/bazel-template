load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

register_toolchains("//toolchains/gcc:cc_toolchain")

new_local_repository(
    name = "cuda",
    build_file = "//third_party/cuda:BUILD",
    path = "/usr/local/cuda",
)

http_archive(
    name = "gtest",
    strip_prefix = "googletest-release-1.12.1",
    urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip"],
)
