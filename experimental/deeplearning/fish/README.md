# Project Fish

Build:

```sh
$ bazel build //experimental/deeplearning/fish/...
```

Export legacy DeepSmith data:

```sh
$ bazel-bin/experimental/deeplearning/fish/export_clang_opencl_dataset \
    --export_path ~/data/experimental/deeplearning/fish/75k
```

Prepare a training set:

```sh
$ bazel-bin/experimental/deeplearning/fish/prepare_discriminator_dataset \
    --export_path ~/data/experimental/deeplearning/fish/75k \
    --dataset_root ~/data/experimental/deeplearning/fish/assertion_dataset \
    --assertions_only
```

Train a discriminator:

```sh
$ bazel-bin/experimental/deeplearning/fish/train_discriminator \
    --dataset_root ~/data/experimental/deeplearning/fish/assertion_dataset  \
    --model_path ~/data/experimental/deeplearning/fish/assertion_model \
    --seed 0
```

Evaluate the discriminator:

```sh
$ TODO(cec):
```
