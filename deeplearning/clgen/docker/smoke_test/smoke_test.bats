#!/usr/bin/env bats
#
# Test running oclgrind.
#
source labm8/sh/test.sh

CORPUS="$(DataPath phd/deeplearning/clgen/tests/data/tiny/corpus.tar.bz2)"
CONFIG="$(DataPath phd/deeplearning/clgen/tests/data/tiny/config.pbtxt)"

TAR="$(DataPath phd/deeplearning/clgen/docker/clgen.tar)"
IMG="bazel/deeplearning/clgen/docker:clgen"

setup() {
  docker load -i "$TAR"
}

teardown() {
  docker rmi --force "$IMG"
}

@test "run clgen" {
  local workdir="$TEST_TMPDIR/clgen"

  mkdir "$workdir"
  cp "$CORPUS" "$workdir/corpus.tar.bz2"
  cp "$CONFIG" "$workdir/config.pbtxt"

  # Set working directory in config file.
  sed -i 's,working_dir: ".*",working_dir: "/clgen",' "$workdir"/config.pbtxt
  # Reduce the runtime by training for fewer epochs.
  sed -i 's/num_epochs: 32/num_epochs: 2/' "$workdir"/config.pbtxt

  # FIXME(cec): Presently the clang_rewriter doesn't work when testing
  # a host-compiled docker image. For this to work, you must build the docker
  # image from within a docker phd_build container.
  sed -i '/preprocessor: "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers"/d' "$workdir"/config.pbtxt

  # Force a user namespace so that generated cache files aren't owned by root.
  # See: https://github.com/moby/moby/issues/3206#issuecomment-152682860
  docker run --rm -v"$workdir":/clgen -u $(id -u):$(id -g) \
    "$IMG" --min_samples=10
}
