# The C++ connector for PostgreSQL.
# http://pqxx.org/development/libpqxx/
load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

licenses(["notice"])  # BSD

configure_make(
    name = "libpqxx",
    configure_options = [
        "--disable-documentation",
    ],
    lib_source = "@libpqxx//:all",
    linkopts = ["-lpq"],
    make_commands = [
        "make -j8",
        "make install",
    ],
    static_libraries = ["libpqxx.a"],
    visibility = ["//visibility:public"],
)
