# This package provides access to the MNIST dataset.
# http://yann.lecun.com/exdb/mnist/

genrule(
    name = "mnist_train_images",
    srcs = ["@mnist_train_images//file"],
    outs = ["mnist_train_images.data"],
    cmd = "gunzip -cd $(location @mnist_train_images//file) > $@",
    visibility = ["//visibility:public"],
)

genrule(
    name = "mnist_test_images",
    srcs = ["@mnist_test_images//file"],
    outs = ["mnist_test_images.data"],
    cmd = "gunzip -cd $(location @mnist_test_images//file) > $@",
    visibility = ["//visibility:public"],
)

genrule(
    name = "mnist_train_labels",
    srcs = ["@mnist_train_labels//file"],
    outs = ["mnist_train_labels.data"],
    cmd = "gunzip -cd $(location @mnist_train_labels//file) > $@",
    visibility = ["//visibility:public"],
)

genrule(
    name = "mnist_test_labels",
    srcs = ["@mnist_test_labels//file"],
    outs = ["mnist_test_labels.data"],
    cmd = "gunzip -cd $(location @mnist_test_labels//file) > $@",
    visibility = ["//visibility:public"],
)
