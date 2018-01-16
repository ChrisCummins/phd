cc_binary(
    name = 'arrays',
    srcs = ['arrays.cc']
)

cc_binary(
    name =  'benchmark-argument-type',
    srcs = ['benchmark-argument-type.cc'],
    copts = ['-Iexternal/benchmark/include'],
    deps = ['@benchmark//:main'],
    linkopts = ['-pthread'],
)

cc_test(
    name =  'constructors',
    srcs = ['constructors.cc'],
    copts = ['-Iexternal/gtest/include'],
    deps = ['@gtest//:main'],
    size = 'small',
)

cc_binary(
    name = 'fork',
    srcs = ['fork.cc'],
)

cc_binary(
    name = 'functional',
    srcs = ['functional.cc'],
)

cc_binary(
    name = 'hash_map',
    srcs = [
        'hash_map.cc',
        'hash_map.h',
    ],
)

cc_binary(
    name = 'inheritance',
    srcs = ['inheritance.cc'],
)

cc_binary(
    name =  'lambdas',
    srcs = ['lambdas.cc'],
    copts = ['-Iexternal/benchmark/include'],
    deps = ['@benchmark//:main'],
    linkopts = ['-pthread'],
)

cc_test(
    name = 'meta',
    srcs = ['meta.cc'],
    size = 'small',
)

cc_binary(
    name = 'myvector',
    srcs = ['myvector.cc'],
)

cc_binary(
    name = 'size-of',
    srcs = ['size-of.cc'],
)

cc_binary(
    name = 'smart-ptr',
    srcs = ['smart-ptr.cc'],
)

cc_binary(
    name = 'strings',
    srcs = ['strings.cc'],
)

cc_binary(
    name = 'templates',
    srcs = ['templates.cc'],
)

cc_binary(
    name = 'user-input',
    srcs = ['user-input.cc'],
)

cc_binary(
    name = 'value-categories',
    srcs = ['value-categories.cc'],
)
