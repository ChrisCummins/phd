load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

# See: https://github.com/bazelbuild/rules_foreign_cc/issues/337#issuecomment-607873621
# See: https://wiki.openssl.org/index.php/Compilation_and_Installation

configure_make(
    name = "openssl",
    configure_command = "config",
    configure_env_vars =
        select({
            "//:darwin": {
                "AR": "ar",
            },
            "//conditions:default": {},
        }),
    configure_options = [
        "no-shared",
    ] + select({
        "//:darwin": [
            "ARFLAGS=r",
        ],
        "//conditions:default": [],
    }),
    lib_source = "@openssl//:all",
    static_libraries = [
        "libssl.a",
        "libcrypto.a",
    ],
    visibility = ["//visibility:public"],
)
