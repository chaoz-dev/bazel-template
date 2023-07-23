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
    "/usr/lib/gcc/x86_64-linux-gnu/11/include",
    "/usr/local/include",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
]

COMPILER_FLAGS = [
    "-std=c++17",
    "-pthread",
    "-fno-canonical-system-headers",
    "-fno-omit-frame-pointer",
    # General warnings and errors.
    "-Wall",
    # "-Werror",
    "-Wextra",
    "-Wunused",
    # Specific warnings and errors.
    "-Walloc-zero",
    "-Wcast-align=strict",
    "-Wcast-qual",
    "-Wconversion",
    "-Wdouble-promotion",
    "-Wduplicated-branches",
    "-Wduplicated-cond",
    "-Wfloat-equal",
    "-Wformat=2",
    "-Wmultiple-inheritance",
    "-Woverloaded-virtual",
    "-Wpointer-arith",
    "-Wshadow",
    "-Wstringop-truncation",
    "-Wsuggest-final-methods",
    "-Wsuggest-final-types",
    "-Wsuggest-override",
    "-Wuninitialized",
    "-Wunreachable-code",
    "-Wwrite-strings",
]

LINKER_FLAGS = [
    "-pthread",
    "-lstdc++",
    "-lm",
    "-lpthread",
]

TOOL_PATHS = [
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
        name = "gcc",
        path = "/usr/bin/gcc",
    ),
    tool_path(
        name = "gcov",
        path = "/usr/bin/gcov",
    ),
    tool_path(
        name = "ld",
        path = "/usr/bin/ld",
    ),
    tool_path(
        name = "nm",
        path = "/usr/bin/nm",
    ),
    tool_path(
        name = "objcopy",
        path = "/usr/bin/objcopy",
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
