load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load(":clang.bzl", "clang_toolchain_config")

LINUX_X86_64 = [
    "@platforms//os:linux",
    "@platforms//cpu:x86_64",
]

filegroup(name = "none")

clang_toolchain_config(name = "clang_toolchain_config")

cc_toolchain(
    name = "clang_toolchain",
    all_files = ":none",
    compiler_files = ":none",
    dwp_files = ":none",
    linker_files = ":none",
    objcopy_files = ":none",
    strip_files = ":none",
    toolchain_config = ":clang_toolchain_config",
    toolchain_identifier = "clang-toolchain",
)

toolchain(
    name = "cc_toolchain",
    exec_compatible_with = LINUX_X86_64,
    target_compatible_with = LINUX_X86_64,
    toolchain = "clang_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
