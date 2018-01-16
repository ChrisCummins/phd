cc_test(
    name = 'main',
    srcs = [
        'main.cc',
        'container.hpp',
    ],
    copts = [
        '-Isrc/phd/include',
        '-Iexternal/gtest/include',
        '-Iexternal/benchmark/include',
    ],
    deps = ['//src/phd:main'],
    size = 'small',
)
