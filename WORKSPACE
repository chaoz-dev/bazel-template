load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# register_toolchains("//toolchains/gcc:cc_toolchain")
register_toolchains("//toolchains/llvm:cc_toolchain")

# Needed for glog.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.zip"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "1865da8c71dbdf89a8c6bc40cac9ea3fd01895e607d6a5f2d97643ca5b3a9412",
    strip_prefix = "glog-chaoz-fork-v0.6.0",
    urls = ["https://github.com/chaoz-dev/glog/archive/chaoz/fork/v0.6.0.zip"],
)

http_archive(
    name = "com_github_google_googletest",
    sha256 = "36ee3f0c8a6760db72d3ef79fe846c1c9120ed17884b46961fc9484f51717c88",
    strip_prefix = "googletest-chaoz-fork-v1.13.0",
    urls = ["https://github.com/chaoz-dev/googletest/archive/chaoz/fork/v1.13.0.zip"],
)

new_local_repository(
    name = "cuda",
    build_file = "//third_party/cuda:BUILD",
    path = "/usr/local/cuda",
)
