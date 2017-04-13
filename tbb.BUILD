cc_library(
    name = 'main',
    srcs = glob(
        ['src/**/*.cpp'],
        exclude = [
            'src/**/test/*.cpp'
        ],
    ),
    hdrs = glob([
        'include/**/*.h',
        'src/tbb/*.h',
        'include/tbb/compat/*',
    ]),
    copts = [
        '-Iexternal/tbb/include',
        '-Iexternal/tbb/include/tbb',
    ],
    linkopts = ['-pthread'],
    visibility = ['//visibility:public'],
)
