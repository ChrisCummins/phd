# Docker image for CLgen

From this directory, run:

```sh
$ ./make_image.sh
```

This produces a docker image `clgen`, with a pre-compiled CLgen build installed
to:

```
/clgen/bin      # CLgen binaries.
/clgen/cache    # Output CLgen cache directory.
/datasets/tiny  # The CLgen tiny corpus.
```

Train and sample the tiny model using:

```sh
# clgen --config /datasets/config.pbtxt
```
