# The C++ connector for PostgreSQL.
# http://pqxx.org/development/libpqxx/
load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

licenses(["notice"])  # BSD

configure_make(
    name = "libpqxx",
    configure_env_vars = select({
        # Workaround for incorrect linker invocation when building on macOS.
        "//:darwin": {"AR": ""},
        "//conditions:default": {},
    }),
    configure_options = [
        "--disable-documentation",
    ],
    lib_source = "@libpqxx//:all",
    linkopts = ["-lpq"],
    static_libraries = ["libpqxx.a"],
    visibility = ["//visibility:public"],
)
