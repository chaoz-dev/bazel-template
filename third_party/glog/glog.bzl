def glog():
    native.local_repository(
        name = "gflags",
        path = "third_party/glog/gflags",
    )
    native.local_repository(
        name = "glog",
        path = "third_party/glog/glog",
        repo_mapping = {
            "@com_github_gflags_gflags": "@gflags",
        },
    )
