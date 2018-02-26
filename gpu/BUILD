# cldrive

sh_library(
    name = "cldrive_src",
    srcs = glob([
        "cldrive/bin/cldrive",
        "cldrive/cldrive/**/*.py",
        "cldrive/Makefile",
        "cldrive/pytest.ini",
        "cldrive/requirements.txt",
        "cldrive/setup.cfg",
        "cldrive/setup.py",
        "cldrive/tests/**/*.py",
        "cldrive/tests/data/**/*",
    ]),
    visibility = ["//visibility:public"],
)

# TODO(cec): Enable cldrive testing once the tests are stable:
# sh_test(
#     name = 'cldrive_test',
#     srcs = ['cldrive/tests/.runner.sh'],
#     args = ['src/cldrive', 'python3.6'],
#     deps = [':cldrive_src'],
#     timeout = 'eternal',
#     size = 'enormous',
# )
