#!/usr/bin/env bash
# A script to export the labelled graph datasets.

TIMESTAMP="$(date '+%Y%m%d')"
EXPORT="$HOME/Inbox/db"
set -eu

mkdir -p "$EXPORT"

create_tarballs() {
    cd "$EXPORT"
    for file in $(ls *.db); do
      echo "$file.tar.bz2"
      tar cjvf "$file.tar.bz2" "$file"
      rm "$file"
    done
    cd -
}

if [[ -f "$EXPORT/devmap_amd_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/devmap_amd_$TIMESTAMP.db"
else
    rm -f "$EXPORT/devmap_amd_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_amd?charset=utf8' \
        --output_db="sqlite:////$EXPORT/devmap_amd_$TIMESTAMP.db"
    create_tarballs
fi

if [[ -f "$EXPORT/devmap_nvidia_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/devmap_nvidia_$TIMESTAMP.db"
else
    rm -f "$EXPORT/devmap_nvidia_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_nvidia?charset=utf8' \
        --output_db="sqlite:////$EXPORT/devmap_nvidia_$TIMESTAMP.db"
    create_tarballs
fi

if [[ -f "$EXPORT/reachability_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/reachability_$TIMESTAMP.db"
else
    rm -f "$EXPORT/reachability_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_reachability?charset=utf8' \
        --output_db="sqlite:////$EXPORT/reachability_$TIMESTAMP.db" \
        --max_rows=1024
    create_tarballs
fi

if [[ -f "$EXPORT/domtree_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/domtree_$TIMESTAMP.db"
else
    rm -f "$EXPORT/domtree_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_domtree?charset=utf8' \
        --output_db="sqlite:////$EXPORT/domtree_$TIMESTAMP.db" \
        --max_rows=1024
    create_tarballs
fi

if [[ -f "$EXPORT/datadep_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/datadep_$TIMESTAMP.db"
else
    rm -f "$EXPORT/datadep_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db="file:///var/phd/db/cc1.mysql?ml4pl_datadep?charset=utf8" \
        --output_db="sqlite:////$EXPORT/datadep_$TIMESTAMP.db" \
        --max_rows=1024
    create_tarballs
fi

if [[ -f "$EXPORT/liveness_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/liveness_$TIMESTAMP.db"
else
    rm -f "$EXPORT/liveness_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db="file:///var/phd/db/cc1.mysql?ml4pl_liveness?charset=utf8" \
        --output_db="sqlite:////$EXPORT/liveness_$TIMESTAMP.db" \
        --max_rows=1024
    create_tarballs
fi
