def cuda():
    native.new_local_repository(
        name = "cuda",
        build_file = "//third_party/cuda:BUILD",
        path = "/usr/local/cuda-12.1",
    )
