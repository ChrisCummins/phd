sh_library(
    name = 'cldrive',
    srcs = glob([
        'cldrive/Makefile',
        'cldrive/requirements.txt',
        'cldrive/setup.py',
        'cldrive/bin/*',
        'cldrive/**/*.py',
    ])
)

sh_test(
    name = 'cldrive-tests',
    srcs = ['cldrive-test.sh'],
    args = [],
    deps = [
        ':cldrive',
    ],
)

sh_library(
    name = 'clgen',
    srcs = glob([
        'clgen/**/*',
    ])
)

sh_test(
    name = 'clgen-tests',
    srcs = ['clgen-test.sh'],
    args = [],
    deps = [
        ':clgen',
    ],
    timeout = 'eternal',
)
