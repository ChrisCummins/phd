#!/usr/bin/env bash
# A script to export the labelled graph datasets.

TIMESTAMP="$(date '+%Y%m%d')"
EXPORT="$HOME/Inbox/db"
set -eu

mkdir -p "$EXPORT"

if [[ -f "$EXPORT/devmap_amd_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/devmap_amd_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_amd_?charset=utf8' \
        --output_db="sqlite:////$EXPORT/devmap_amd_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

if [[ -f "$EXPORT/devmap_nvidia_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/devmap_nvidia_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_devmap_nvidia_?charset=utf8' \
        --output_db="sqlite:////$EXPORT/devmap_nvidia_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

if [[ -f "$EXPORT/reachability_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/reachability_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_reachability_?charset=utf8' \
        --output_db="sqlite:////$EXPORT/reachability_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

if [[ -f "$EXPORT/domtree_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/domtree_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_domtree_?charset=utf8' \
        --output_db="sqlite:////$EXPORT/domtree_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

if [[ -f "$EXPORT/datadep_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/datadep_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
        --input_db="file:///var/phd/db/cc1.mysql?ml4pl_datadep_?charset=utf8" \
        --output_db="sqlite:////$EXPORT/datadep_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

if [[ -f "$EXPORT/liveness_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/liveness_$TIMESTAMP.db"
else
    bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
        --input_db="file:///var/phd/db/cc1.mysql?ml4pl_liveness_?charset=utf8" \
        --output_db="sqlite:////$EXPORT/liveness_$TIMESTAMP.db" \
        --alsologtostderr \
        --max_rows=1024
fi

cd "$EXPORT"
for file in $(ls *.db); do
  echo "$file.tar.bz2"
  tar cjvf "$file.tar.bz2" "$file"
  rm "$file"
done
