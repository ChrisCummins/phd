# This package contains additional tests for me.db. These are higher level tests
# than the unit tests that are stored inline with the source packages.

# Real-world data that has been obfuscated and anonymized.
filegroup(
    name = "test_inbox_files",
    # testonly = 1,
    srcs = glob(["test_inbox/**"]),
    visibility = ["//datasets/me_db/tests:__subpackages__"],
)
