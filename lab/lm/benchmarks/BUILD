cc_test(
    name = 'main',
    srcs = [
        'benchmarks.cc',
        'benchmarks.hpp',
    ],
    copts = [
        '-Ilab/lm/include',
        '-Isrc/phd/include',
        '-Iexternal/gtest/include',
        '-Iexternal/benchmark/include',
    ],
    deps = [
        '//lab/lm:main',
        '//src/phd:main',
    ],
    size = 'small',
)
