load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def torch():
    http_archive(
        name = "org_pytorch_libtorch",
        urls = ["https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"],
        sha256 = "fc1afc2fb28e2ab7b526df04c574c173fc6f068b5fbca831734607ef39fd6893",
        build_file = "//third_party/torch:BUILD",
    )

    native.new_local_repository(
        name = "omp",
        build_file = "//third_party/torch:BUILD",
        path = "/usr/lib/x86_64-linux-gnu",
    )

    native.new_local_repository(
        name = "python",
        build_file = "//third_party/torch:BUILD",
        path = "/usr/lib/x86_64-linux-gnu",
    )
