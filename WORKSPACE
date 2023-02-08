load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# register_toolchains("//toolchains/gcc:cc_toolchain")
register_toolchains("//toolchains/llvm:cc_toolchain")

new_local_repository(
    name = "cuda",
    build_file = "//third_party/cuda:BUILD",
    path = "/usr/local/cuda",
)

# Needed for glog.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    strip_prefix = "glog-0.6.0",
    urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
)

http_archive(
    name = "com_github_google_googletest",
    strip_prefix = "googletest-release-1.12.1",
    urls = ["https://github.com/google/googletest/archive/release-1.12.1.zip"],
)
