load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

INCLUDE_DIRS = [
    "/usr/include/c++/11",
    "/usr/include/x86_64-linux-gnu/c++/11",
    "/usr/include/c++/11/backward",
    "/usr/lib/llvm-14/include/c++/v1/",
    "/usr/lib/llvm-14/lib/clang/14.0.0/include",
    "/usr/local/include",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
]

COMPILER_FLAGS = [
    "-std=c++17",
    "-stdlib=libc++",
    "-fno-omit-frame-pointer",
    # General warnings and errors.
    "-Wall",
    "-Werror",
    # Specific warnings and errors.
    "-Wcast-align",
    "-Wcast-qual",
    "-Wconversion",
    "-Wfloat-equal",
    "-Wformat=2",
    "-Wpointer-arith",
    "-Wshadow",
    "-Wuninitialized",
    "-Wunreachable-code",
    "-Wunused-but-set-parameter",
    "-Wwrite-strings",
    # Disabled warnings and errors.
    "-Wno-free-nonheap-object",
    "-Wno-noexcept-type",
]

LINKER_FLAGS = [
    "-fuse-ld=lld",
    "-rtlib=compiler-rt",
    "-lc++",
    "-lc++abi",
    "-lm",
]

BIN_PATH = "/usr/lib/llvm-14/bin/"
TOOL_PATHS = [
    tool_path(
        name = "ar",
        path = BIN_PATH + "llvm-ar",
    ),
    tool_path(
        name = "as",
        path = BIN_PATH + "llvm-as",
    ),
    tool_path(
        name = "cpp",
        path = BIN_PATH + "clang-cpp",
    ),
    tool_path(
        name = "gcc",
        path = BIN_PATH + "clang",
    ),
    tool_path(
        name = "gcov",
        path = BIN_PATH + "llvm-cov",
    ),
    tool_path(
        name = "ld",
        path = BIN_PATH + "lld",
    ),
    tool_path(
        name = "nm",
        path = BIN_PATH + "llvm-nm",
    ),
    tool_path(
        name = "objcopy",
        path = BIN_PATH + "llvm-objcopy",
    ),
    tool_path(
        name = "objdump",
        path = BIN_PATH + "llvm-objdump",
    ),
    tool_path(
        name = "strip",
        path = BIN_PATH + "llvm-strip",
    ),
]

COMPILER_FLAGS_FEATURE = feature(
    name = "compiler_flags",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.cpp_compile],
            flag_groups = [
                flag_group(flags = COMPILER_FLAGS),
            ],
        ),
    ],
)

LINKER_FLAGS_FEATURE = feature(
    name = "linker_flags",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [
                ACTION_NAMES.cpp_link_executable,
                ACTION_NAMES.cpp_link_dynamic_library,
                ACTION_NAMES.cpp_link_nodeps_dynamic_library,
            ],
            flag_groups = [
                flag_group(flags = LINKER_FLAGS),
            ],
        ),
    ],
)

def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        compiler = "clang",
        ctx = ctx,
        cxx_builtin_include_directories = INCLUDE_DIRS,
        features = [COMPILER_FLAGS_FEATURE, LINKER_FLAGS_FEATURE],
        host_system_name = "local",
        target_cpu = "k8",
        target_libc = "local",
        target_system_name = "local",
        tool_paths = TOOL_PATHS,
        toolchain_identifier = "local",
    )

clang_toolchain_config = rule(
    provides = [CcToolchainConfigInfo],
    implementation = _impl,
)
