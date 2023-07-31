load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def gtest():
    http_archive(
        name = "com_github_google_googletest",
        urls = ["https://github.com/google/googletest/archive/v1.13.0.zip"],
        sha256 = "ffa17fbc5953900994e2deec164bb8949879ea09b411e07f215bfbb1f87f4632",
        strip_prefix = "googletest-1.13.0",
        patches = [
            "//third_party/gtest/patches:0001-googletest-BUILD-Ignore-compile-warnings.patch",
        ],
        patch_args = ["-p1"],
    )
