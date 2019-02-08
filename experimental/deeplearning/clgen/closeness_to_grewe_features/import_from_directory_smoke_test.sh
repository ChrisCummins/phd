#!/usr/bin/env bash
#
# Test that `import_from_directory` runs without catching fire.
#
set -eux

TMPDIR="/tmp/phd/clgen"

mkdir -pv $TMPDIR/kernels

cat <<EOF > "$TMPDIR/kernels/a.cl"
kernel void A(global int* a) {
  if (get_global_id(0) < 1000000) {
    a[get_global_id(0)] = 0;
  }
}
EOF

experimental/deeplearning/clgen/closeness_to_grewe_features/import_from_directory \
    --db="sqlite:///$TMPDIR/db" \
    --kernels_dir="$TMPDIR/kernels" \
    --origin=clgen

# Run it again to see if it fails when processing duplicate data.
experimental/deeplearning/clgen/closeness_to_grewe_features/import_from_directory \
    --db="sqlite:///$TMPDIR/db" \
    --kernels_dir="$TMPDIR/kernels" \
    --origin=clgen

test -f "$TMPDIR/db"

rm -rf "$TMPDIR"
