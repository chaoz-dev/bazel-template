load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def glog():
    http_archive(
        name = "com_github_gflags_gflags",
        urls = ["https://github.com/gflags/gflags/archive/v2.2.2.zip"],
        sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
        strip_prefix = "gflags-2.2.2",
        patches = [
            "//third_party/glog/patches:0001-gflags-BUILD-Ignore-compile-warnings.patch",
        ],
        patch_args = ["-p1"],
    )
    http_archive(
        name = "com_github_google_glog",
        urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
        sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
        strip_prefix = "glog-0.6.0",
        patches = [
            "//third_party/glog/patches:0001-glog-BUILD-Ignore-compile-warnings.patch",
        ],
        patch_args = ["-p1"],
    )
