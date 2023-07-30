load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//rules:nvcc.bzl", "nvcc_binary", "nvcc_library")

nvcc_library(
    name = "print_kernel_ids",
    srcs = ["print_kernel_ids.cu.cc"],
    hdrs = ["print_kernel_ids.hh"],
)

# We can depend on an `nvcc_library` from any `cc_library` or `cc_binary`.
cc_binary(
    name = "print_kernel_ids_cc",
    srcs = ["print_kernel_ids.cc"],
    deps = [":print_kernel_ids"],
)

# We can also run the `.cu.cc` file as an `nvcc_binary` if it defines a `main` function.
nvcc_binary(
    name = "print_kernel_ids_nvcc",
    srcs = ["print_kernel_ids.cu.cc"],
)
