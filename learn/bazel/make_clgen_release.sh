# GOAL: Make a standalone binary release of CLgen for macOS and linux.

# Requirements: python3.6
# TODO: pip packages
# TODO: homebrew llvm, until config.pbtxt is dropped.

bazel build //deeplearning/clgen
cd bazel-phd/bazel-out/darwin-py3-opt/bin/deeplearning/clgen
tar cjvfh clgen_darwn.tar.bz2 \
  --exclude '*.runfiles_manifest' \
  --exclude '*.intellij-info.txt' \
  --exclude 'MANIFEST' \
  --exclude '__pycache__' \
  clgen_test clgen_test.runfiles
