# Project Fish

Build:

```sh
$ bazel build //experimental/deeplearning/fish/...
```

Export data to train a discriminator:

```sh
$ bazel-bin/experimental/deeplearning/fish/export_discriminator_training_set \
  --export_path ~/data/experimental/deeplearning/fish/75k
```

Train a discriminator:

```sh
$ TODO(cec): bazel-bin/experimental/deeplearning/fish/train_discriminator
```

Evaluate the discriminator:

```sh
$ TODO(cec):
```
