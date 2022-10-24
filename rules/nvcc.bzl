load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME")
load("@rules_cc//cc:defs.bzl", "cc_library")

HDR_FILES = [".h", ".hh", ".hpp", ".inc", ".inl", ".cuh", ".cu.hh"]
SRC_FILES = HDR_FILES + [".c", ".cc", ".cpp", ".cu", ".cu.cc"]

NVCC_COPTS = [
    "--compile",
    "--std=c++17",
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
    host_copts.append("-B/usr/bin/")

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

    nvcc_copts += ctx.attr.nvcc_copts
    return nvcc_copts

def _nvcc_library_impl(ctx):
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

_nvcc_library = rule(
    implementation = _nvcc_library_impl,
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

def nvcc_library(name, srcs = [], hdrs = [], deps = [], host_copts = [], nvcc_copts = []):
    nvcc_lib = "_nvcc_" + name

    _nvcc_library(
        name = nvcc_lib,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + ["@cuda"],
        host_copts = host_copts,
        nvcc_copts = nvcc_copts + NVCC_COPTS,
    )

    cc_library(
        name = name,
        srcs = [nvcc_lib] if srcs else [],
        hdrs = hdrs,
        linkopts = CC_LINKOPTS,
        deps = deps + ["@cuda"],
    )
