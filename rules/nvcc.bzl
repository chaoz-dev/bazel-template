load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

HDR_FILES = [".cu.hh", ".cuh", ".h", ".hh", ".hpp", ".in", ".inc", ".inl"]
SRC_FILES = HDR_FILES + [".c", ".cc", ".cpp", ".cu", ".cu.cc"]

BIN_DIR = "/usr/bin"

COMPUTE_CAPABILITIES = [
    # Volta
    70,
    # Turing
    75,
    # Ampere
    80,
    86,
    # Ada
    89,
    # Hopper
    90,
]

CC_COPTS = [
    "-Wno-old-style-cast",
    "-Wno-reserved-identifier",
    "-Wno-overlength-strings",
]

NVCC_COPTS = [
    "--compile",
    "--std=c++20",
    "--x",
    "cu",
]

CC_LINKOPTS = [
    "-ldl",
    "-lpthread",
]

def _cc_features(ctx):
    return cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = find_cpp_toolchain(ctx),
    )

def _cc_compiler_path(ctx):
    return cc_common.get_tool_for_action(
        feature_configuration = _cc_features(ctx),
        action_name = CPP_COMPILE_ACTION_NAME,
    )

def _host_copts(ctx):
    host_copts = []

    cc_variables = cc_common.create_compile_variables(
        feature_configuration = _cc_features(ctx),
        cc_toolchain = find_cpp_toolchain(ctx),
        user_compile_flags = ctx.fragments.cpp.copts,
        use_pic = True,
    )
    host_copts += cc_common.get_memory_inefficient_command_line(
        feature_configuration = _cc_features(ctx),
        action_name = CPP_COMPILE_ACTION_NAME,
        variables = cc_variables,
    )

    host_copts += ctx.attr.host_copts

    host_copts.append("-B{path}".format(path = BIN_DIR))

    return host_copts

def _nvcc_copts(ctx):
    nvcc_copts = [
        "--compiler-bindir={}".format(_cc_compiler_path(ctx)),
    ]

    includes = depset(
        transitive = [dep[CcInfo].compilation_context.includes for dep in ctx.attr.deps],
    ).to_list()
    defines = depset(
        transitive = [dep[CcInfo].compilation_context.defines for dep in ctx.attr.deps],
    ).to_list()

    nvcc_copts += (
        ["-I."] +
        ["-I{}".format(include) for include in includes] +
        ["-D{}".format(define) for define in defines]
    )

    system_includes = depset(
        transitive = [dep[CcInfo].compilation_context.system_includes for dep in ctx.attr.deps],
    ).to_list()
    quote_includes = depset(
        transitive = [dep[CcInfo].compilation_context.quote_includes for dep in ctx.attr.deps],
    ).to_list()

    for system_include in system_includes:
        nvcc_copts.extend([
            "--compiler-options",
            "-isystem",
            "--compiler-options",
            system_include,
        ])

    for quote_include in quote_includes:
        nvcc_copts.extend([
            "--compiler-options",
            "-iquote",
            "--compiler-options",
            quote_include,
        ])

    for cc_opt in _host_copts(ctx):
        nvcc_copts.extend([
            "--compiler-options",
            cc_opt,
        ])

    for compute_capability in COMPUTE_CAPABILITIES:
        nvcc_copts.extend([
            "-gencode=arch=compute_{},code=compute_{}".format(compute_capability, compute_capability),
            "-gencode=arch=compute_{},code=sm_{}".format(compute_capability, compute_capability),
        ])

    nvcc_copts += ctx.attr.nvcc_copts
    return nvcc_copts

def _nvcc_library_rule_impl(ctx):
    if not ctx.attr.srcs:
        return [DefaultInfo()]

    nvcc_copts = _nvcc_copts(ctx)

    hdr_deps = depset(ctx.files.hdrs)
    other_deps = depset(transitive = [dep[CcInfo].compilation_context.headers for dep in ctx.attr.deps])
    toolchain_deps = depset(ctx.files._nvcc_toolchain, transitive = [find_cpp_toolchain(ctx).all_files])

    outs = []
    for src in ctx.files.srcs:
        out = ctx.actions.declare_file(src.basename.split(".")[0] + ".o", sibling = src)
        ctx.actions.run(
            outputs = [out],
            inputs = depset([src], transitive = [hdr_deps, other_deps, toolchain_deps]),
            executable = ctx.executable._nvcc,
            arguments = nvcc_copts + [
                "--keep",
                "--keep-dir",
                out.dirname,
                src.path,
                "--output-file",
                out.path,
            ],
            env = {"TMPDIR": out.dirname},
        )
        outs.append(out)

    return [DefaultInfo(files = depset(outs))]

_nvcc_library_rule = rule(
    implementation = _nvcc_library_rule_impl,
    fragments = ["cpp"],
    attrs = {
        "srcs": attr.label_list(allow_files = SRC_FILES),
        "hdrs": attr.label_list(allow_files = HDR_FILES),
        "deps": attr.label_list(providers = [CcInfo]),
        "host_copts": attr.string_list(),
        "nvcc_copts": attr.string_list(),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_nvcc": attr.label(
            default = Label("@cuda//:nvcc"),
            executable = True,
            cfg = "exec",
        ),
        "_nvcc_toolchain": attr.label(default = "@cuda//:toolchain"),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)

def _nvcc_library(name, srcs, hdrs, deps, host_copts, nvcc_copts, testonly = False):
    _nvcc_library_rule(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = [
            "//third_party/cuda",
        ] + deps,
        host_copts = CC_COPTS + host_copts,
        nvcc_copts = NVCC_COPTS + nvcc_copts,
        testonly = testonly,
    )

def nvcc_library(name, srcs = [], hdrs = [], deps = [], host_copts = [], nvcc_copts = [], visibility = ["//visibility:private"]):
    nvcc_lib = "_nvcc_" + name

    _nvcc_library(
        name = nvcc_lib,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        host_copts = host_copts,
        nvcc_copts = nvcc_copts,
    )

    cc_library(
        name = name,
        srcs = [nvcc_lib] if srcs else [],
        hdrs = hdrs,
        linkopts = CC_LINKOPTS,
        deps = [
            "//third_party/cuda",
        ] + deps,
        visibility = visibility,
    )

def nvcc_binary(name, srcs = [], hdrs = [], deps = [], host_copts = [], nvcc_copts = []):
    nvcc_lib_name = "_nvcc_" + name

    _nvcc_library(
        name = nvcc_lib_name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        host_copts = [
            "-DNVCC_BINARY",
        ] + host_copts,
        nvcc_copts = nvcc_copts,
    )

    cc_binary(
        name = name,
        srcs = [nvcc_lib_name] if srcs else [],
        linkopts = CC_LINKOPTS,
        deps = [
            "//third_party/cuda",
        ] + deps,
    )

def nvcc_test(name, srcs = [], hdrs = [], deps = [], host_copts = [], nvcc_copts = []):
    nvcc_lib_name = "_nvcc_" + name

    _nvcc_library(
        name = nvcc_lib_name,
        srcs = srcs,
        hdrs = hdrs,
        deps = [
            "//third_party/gtest",
        ] + deps,
        host_copts = host_copts,
        nvcc_copts = nvcc_copts,
        testonly = True,
    )

    cc_test(
        name = name,
        srcs = [nvcc_lib_name] if srcs else [],
        linkopts = CC_LINKOPTS,
        deps = deps + [
            "//third_party/cuda",
            "//third_party/gtest",
        ],
        testonly = True,
    )
