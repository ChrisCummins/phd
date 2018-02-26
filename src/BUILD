# cldrive

sh_library(
    name = 'cldrive_src',
    srcs = glob([
        'cldrive/bin/cldrive',
        'cldrive/cldrive/**/*.py',
        'cldrive/Makefile',
        'cldrive/pytest.ini',
        'cldrive/requirements.txt',
        'cldrive/setup.cfg',
        'cldrive/setup.py',
        'cldrive/tests/**/*.py',
        'cldrive/tests/data/**/*',
    ]),
    visibility = ['//visibility:public'],
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

# clgen

sh_library(
    name = 'clgen_src',
    srcs = glob([
        'clgen/bin/*',
        'clgen/clgen/**/*.py',
        'clgen/clgen/data/**/*',
        'clgen/configure',
        'clgen/make/**/*',
        'clgen/Makefile',
        'clgen/native/*.cpp',
        'clgen/setup.py',
        'clgen/tests/**/*',
    ]),
    visibility = ['//visibility:public'],
)

# a script which sets up a virtualenv and runs the test suite.
# TODO(cec): Enable clgen testing once the tests are stable:
# sh_test(
#     name = 'clgen_test',
#     srcs = ['clgen_runner.sh'],
#     args = ['src/clgen', 'python3.6'],
#     deps = [':clgen_src'],
#     timeout = 'eternal',
#     size = 'enormous',
# )
