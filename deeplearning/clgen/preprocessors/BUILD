# This package defines the preprocessor pipeline for CLgen.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_library(
    name = "clang",
    srcs = ["clang.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
            "@llvm_linux//:clang",
        ],
    }),
    deps = [
        ":public",
        "//compilers/llvm",
        "//compilers/llvm:clang_format",
        "//deeplearning/clgen:errors",
        "//labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    deps = [
        ":clang",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

# The rather ludicrous combination of data, of copts and linkopts is a result
# of bashing my head against the wall for a few days trying to get the damn
# things to compile and link against a remote LLVM binary release. The
# abomination of options below works as of the time of writing (2018-06-12).
# This is *definitely* not the best way to do it, but it's the only way I
# could get it work given the seemingly picky ordering of LLVM link libraries,
# and the sandboxing of bazel's build.
cc_binary(
    name = "clang_rewriter",
    srcs = ["clang_rewriter.cpp"],
    copts = [
        "-fPIC",
        "-fvisibility-inlines-hidden",
        "-Werror=date-time",
        "-std=c++11",
        "-Wall",
        "-W",
        "-Wno-unused-parameter",
        "-Wwrite-strings",
        "-Wcast-qual",
        "-Wno-missing-field-initializers",
        "-pedantic",
        "-Wno-long-long",
        "-Wdelete-non-virtual-dtor",
        "-Wno-comment",
        "-fno-rtti",
        "-ffunction-sections",
        "-fdata-sections",
        "-DNDEBUG",
        "-D_GNU_SOURCE",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_FORMAT_MACROS",
        "-D__STDC_LIMIT_MACROS",
    ],
    data = select({
        "//:darwin": ["@llvm_mac//:libdir"],
        "//conditions:default": ["@llvm_linux//:libdir"],
    }),
    linkopts = select({
        "//:darwin": [
            "-Wl,-rpath,$(location @llvm_mac//:libdir)/",
            "-L$(location @llvm_mac//:libdir)/",
        ],
        "//conditions:default": [
            "-Wl,-rpath,$(location @llvm_linux//:libdir)/",
            "-L$(location @llvm_linux//:libdir)/",
        ],
    }) + [
        # TODO(cec): I expect that this list can be pruned somewhat.
        "-pthread",
        "-ldl",
        "-lz",
        "-lclangARCMigrate",
        "-lclangFrontend",
        "-lclangFrontendTool",
        "-lclangDriver",
        "-lclangSerialization",
        "-lclangCodeGen",
        "-lclangParse",
        "-lclangSema",
        "-lclangRewriteFrontend",
        "-lclangRewrite",
        "-lclangStaticAnalyzerFrontend",
        "-lclangStaticAnalyzerCheckers",
        "-lclangStaticAnalyzerCore",
        "-lclangAnalysis",
        "-lclangEdit",
        "-lclangAST",
        "-lclangASTMatchers",
        "-lclangLex",
        "-lclangBasic",
        "-lclangTooling",
        "-lclangToolingCore",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["-lclang"],
    }) + [
        "-lLLVMLTO",
        "-lLLVMPasses",
        "-lLLVMObjCARCOpts",
        "-lLLVMMIRParser",
        "-lLLVMSymbolize",
        "-lLLVMDebugInfoPDB",
        "-lLLVMDebugInfoDWARF",
        "-lLLVMCoverage",
        "-lLLVMTableGen",
        "-lLLVMDlltoolDriver",
        "-lLLVMOrcJIT",
        "-lLLVMXCoreDisassembler",
        "-lLLVMXCoreCodeGen",
        "-lLLVMXCoreDesc",
        "-lLLVMXCoreInfo",
        "-lLLVMXCoreAsmPrinter",
        "-lLLVMSystemZDisassembler",
        "-lLLVMSystemZCodeGen",
        "-lLLVMSystemZAsmParser",
        "-lLLVMSystemZDesc",
        "-lLLVMSystemZInfo",
        "-lLLVMSystemZAsmPrinter",
        "-lLLVMSparcDisassembler",
        "-lLLVMSparcCodeGen",
        "-lLLVMSparcAsmParser",
        "-lLLVMSparcDesc",
        "-lLLVMSparcInfo",
        "-lLLVMSparcAsmPrinter",
        "-lLLVMPowerPCDisassembler",
        "-lLLVMPowerPCCodeGen",
        "-lLLVMPowerPCAsmParser",
        "-lLLVMPowerPCDesc",
        "-lLLVMPowerPCInfo",
        "-lLLVMPowerPCAsmPrinter",
        "-lLLVMNVPTXCodeGen",
        "-lLLVMNVPTXDesc",
        "-lLLVMNVPTXInfo",
        "-lLLVMNVPTXAsmPrinter",
        "-lLLVMMSP430CodeGen",
        "-lLLVMMSP430Desc",
        "-lLLVMMSP430Info",
        "-lLLVMMSP430AsmPrinter",
        "-lLLVMMipsDisassembler",
        "-lLLVMMipsCodeGen",
        "-lLLVMMipsAsmParser",
        "-lLLVMMipsDesc",
        "-lLLVMMipsInfo",
        "-lLLVMMipsAsmPrinter",
        "-lLLVMLanaiDisassembler",
        "-lLLVMLanaiCodeGen",
        "-lLLVMLanaiAsmParser",
        "-lLLVMLanaiDesc",
        "-lLLVMLanaiAsmPrinter",
        "-lLLVMLanaiInfo",
        "-lLLVMHexagonDisassembler",
        "-lLLVMHexagonCodeGen",
        "-lLLVMHexagonAsmParser",
        "-lLLVMHexagonDesc",
        "-lLLVMHexagonInfo",
        "-lLLVMBPFDisassembler",
        "-lLLVMBPFCodeGen",
        "-lLLVMBPFAsmParser",
        "-lLLVMBPFDesc",
        "-lLLVMBPFInfo",
        "-lLLVMBPFAsmPrinter",
        "-lLLVMARMDisassembler",
        "-lLLVMARMCodeGen",
        "-lLLVMARMAsmParser",
        "-lLLVMARMDesc",
        "-lLLVMARMInfo",
        "-lLLVMARMAsmPrinter",
        "-lLLVMARMUtils",
        "-lLLVMAMDGPUDisassembler",
        "-lLLVMAMDGPUCodeGen",
        "-lLLVMAMDGPUAsmParser",
        "-lLLVMAMDGPUDesc",
        "-lLLVMAMDGPUInfo",
        "-lLLVMAMDGPUAsmPrinter",
        "-lLLVMAMDGPUUtils",
        "-lLLVMAArch64Disassembler",
        "-lLLVMAArch64CodeGen",
        "-lLLVMAArch64AsmParser",
        "-lLLVMAArch64Desc",
        "-lLLVMAArch64Info",
        "-lLLVMAArch64AsmPrinter",
        "-lLLVMAArch64Utils",
        "-lLLVMObjectYAML",
        "-lLLVMLibDriver",
        "-lLLVMOption",
        "-lLLVMWindowsManifest",
        "-lLLVMFuzzMutate",
        "-lLLVMX86Disassembler",
        "-lLLVMX86AsmParser",
        "-lLLVMX86CodeGen",
        "-lLLVMGlobalISel",
        "-lLLVMSelectionDAG",
        "-lLLVMAsmPrinter",
        "-lLLVMDebugInfoCodeView",
        "-lLLVMDebugInfoMSF",
        "-lLLVMX86Desc",
        "-lLLVMMCDisassembler",
        "-lLLVMX86Info",
        "-lLLVMX86AsmPrinter",
        "-lLLVMX86Utils",
        "-lLLVMMCJIT",
        "-lLLVMLineEditor",
        "-lLLVMInterpreter",
        "-lLLVMExecutionEngine",
        "-lLLVMRuntimeDyld",
        "-lLLVMCodeGen",
        "-lLLVMTarget",
        "-lLLVMCoroutines",
        "-lLLVMipo",
        "-lLLVMInstrumentation",
        "-lLLVMVectorize",
        "-lLLVMScalarOpts",
        "-lLLVMLinker",
        "-lLLVMIRReader",
        "-lLLVMAsmParser",
        "-lLLVMInstCombine",
        "-lLLVMTransformUtils",
        "-lLLVMBitWriter",
        "-lLLVMAnalysis",
        "-lLLVMProfileData",
        "-lLLVMObject",
        "-lLLVMMCParser",
        "-lLLVMMC",
        "-lLLVMBitReader",
        "-lLLVMCore",
        "-lLLVMBinaryFormat",
        "-lLLVMSupport",
        "-lLLVMDemangle",
    ] + select({
        "//:darwin": ["-lncurses"],
        "//conditions:default": [],
    }),
    deps = select({
        "//:darwin": [
            "@llvm_headers_mac//:headers",
            "@llvm_mac//:libs",
        ],
        "//conditions:default": [
            "@llvm_headers_linux//:headers",
            "@llvm_linux//:libs",
        ],
    }),
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":public",
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "cxx",
    srcs = ["cxx.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
        ],
    }),
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//compilers/llvm:clang",
        "//labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "cxx_test",
    srcs = ["cxx_test.py"],
    deps = [
        ":cxx",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

# FIXME(cec): py3_image is not a test rule, but //labm8:test is marked
# test-only.
# py3_image(
#     name = "cxx_test_image",
#     srcs = ["cxx_test.py"],
#     base = "@base//image",
#     main = "cxx_test.py",
#     deps = [
#         ":cxx",
#         "//deeplearning/clgen:errors",
#         "//labm8:test",
#         "//third_party/py/absl",
#     ],
# )

py_library(
    name = "java",
    srcs = ["java.py"],
    data = [":JavaRewriter"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        ":public",
        "//deeplearning/clgen:errors",
        "//labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "java_test",
    srcs = ["java_test.py"],
    deps = [
        ":java",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

java_binary(
    name = "JavaRewriter",
    srcs = ["JavaRewriter.java"],
    main_class = "deeplearning.clgen.preprocessors.JavaRewriter",
    deps = [
        "@com_google_guava//jar",
        "@org_eclipse_core_contenttype//jar",
        "@org_eclipse_core_runtime//jar",
        "@org_eclipse_equinox_common//jar",
        "@org_eclipse_equinox_preferences//jar",
        "@org_eclipse_jdt_core//jar",
        "@org_eclipse_jface//jar",
        "@org_eclipse_jobs//jar",
        "@org_eclipse_ltk_core_refactoring//jar",
        "@org_eclipse_ltk_ui_refactoring//jar",
        "@org_eclipse_osgi//jar",
        "@org_eclipse_osgi_util//jar",
        "@org_eclipse_resources//jar",
        "@org_eclipse_text//jar",
        "@org_osgi_framework//jar",
        "@org_osgi_service_prefs//jar",
    ],
)

py_library(
    name = "normalizer",
    srcs = ["normalizer.py"],
    data = [
        ":clang_rewriter",
    ],
    deps = [
        "//deeplearning/clgen:errors",
        "//labm8:bazelutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "normalizer_test",
    srcs = ["normalizer_test.py"],
    deps = [
        ":normalizer",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    data = [
        "//deeplearning/clgen/data/include:opencl",
        "//deeplearning/clgen/data/include:opencl-shim",
        "//third_party/libclc/generic/include:headers",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":clang",
        ":normalizer",
        ":public",
        "//labm8:bazelutil",
    ],
)

py_test(
    name = "opencl_test",
    srcs = ["opencl_test.py"],
    data = ["//deeplearning/clgen/data/include:opencl-shim"],
    deps = [
        ":clang",
        ":opencl",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//labm8:bazelutil",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//visibility:public"],
    deps = [
        # This target *must* depend on all modules which expose a
        # @public.clgen_preprocessor decorated function, since the method
        # GetPreprocessorFunction() will try to import it. Otherwise it would
        # be the responsibility of any code which calls into one of the public
        # PreprocessX() functions to ensure that the module which implements
        # each preprocessor is listed in its deps.
        ":common",
        ":cxx",
        ":java",
        ":normalizer",
        ":opencl",
        ":public",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "preprocessors_test",
    srcs = ["preprocessors_test.py"],
    deps = [
        ":preprocessors",
        ":public",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "public_test",
    srcs = ["public_test.py"],
    deps = [
        ":public",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
