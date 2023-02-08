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
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.zip"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
)

http_archive(
    name = "com_github_google_googletest",
    sha256 = "24564e3b712d3eb30ac9a85d92f7d720f60cc0173730ac166f27dda7fed76cb2",
    strip_prefix = "googletest-release-1.12.1",
    urls = ["https://github.com/google/googletest/archive/release-1.12.1.zip"],
)
