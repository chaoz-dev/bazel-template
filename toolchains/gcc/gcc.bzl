load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

INCLUDE_DIRS = [
    "/usr/include",
    "/usr/include/x86_64-linux-gnu",
    "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    "/usr/local/include",
]

COMPILER_FLAGS = [
    "-std=c++17",
    "-fno-omit-frame-pointer",
    "-fno-canonical-system-headers",
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
    "-lstdc++",
    "-lm",
]

TOOL_PATHS = [
    tool_path(
        name = "gcc",
        path = "/usr/bin/gcc",
    ),
    tool_path(
        name = "ld",
        path = "/usr/bin/ld",
    ),
    tool_path(
        name = "ar",
        path = "/usr/bin/ar",
    ),
    tool_path(
        name = "as",
        path = "/usr/bin/as",
    ),
    tool_path(
        name = "cpp",
        path = "/usr/bin/cpp",
    ),
    tool_path(
        name = "gcov",
        path = "/usr/bin/gcov",
    ),
    tool_path(
        name = "nm",
        path = "/usr/bin/nm",
    ),
    tool_path(
        name = "objdump",
        path = "/usr/bin/objdump",
    ),
    tool_path(
        name = "strip",
        path = "/usr/bin/strip",
    ),
]

COMPILER_FLAGS_FEATURE = feature(
    name = "compiler_flags",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = [ACTION_NAMES.cpp_compile],
            flag_groups = ([
                flag_group(flags = COMPILER_FLAGS),
            ]),
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
            flag_groups = ([
                flag_group(flags = LINKER_FLAGS),
            ]),
        ),
    ],
)

def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = [COMPILER_FLAGS_FEATURE, LINKER_FLAGS_FEATURE],
        toolchain_identifier = "local",
        cxx_builtin_include_directories = INCLUDE_DIRS,
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "local",
        compiler = "gcc",
        tool_paths = TOOL_PATHS,
    )

gcc_toolchain_config = rule(
    implementation = _impl,
    provides = [CcToolchainConfigInfo],
)
