load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load(":llvm.bzl", "llvm_toolchain_config")

LINUX_X86_64 = [
    "@platforms//os:linux",
    "@platforms//cpu:x86_64",
]

filegroup(name = "none")

llvm_toolchain_config(name = "llvm_toolchain_config")

cc_toolchain(
    name = "llvm_toolchain",
    all_files = ":none",
    compiler_files = ":none",
    dwp_files = ":none",
    linker_files = ":none",
    objcopy_files = ":none",
    strip_files = ":none",
    toolchain_config = ":llvm_toolchain_config",
    toolchain_identifier = "llvm-toolchain",
)

toolchain(
    name = "cc_toolchain",
    exec_compatible_with = LINUX_X86_64,
    target_compatible_with = LINUX_X86_64,
    toolchain = "llvm_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
