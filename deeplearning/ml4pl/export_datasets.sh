#!/usr/bin/env bash
# A script to export the labelled graph datasets.

TIMESTAMP="$(date '+%Y%m%d')"
EXPORT="$HOME/db"
set -eu

mkdir -p "$EXPORT"

create_tarballs() {
    cd "$EXPORT"
    for file in $(ls *.db); do
      echo "$file.tar.bz2"
      tar cjvf "$file.tar.bz2" "$file"
      # rm "$file"
    done
    cd -
}

export_dataset() {
  local dataset="$1"
  echo "exporting ${dataset} ..."
  if [[ -f "$EXPORT/${dataset}_$TIMESTAMP.db.tar.bz2" ]]; then
    echo "Skipping $EXPORT/${dataset}_$TIMESTAMP.db"
  else
    rm -f "$EXPORT/${dataset}_$TIMESTAMP.db"
    bazel run //deeplearning/ml4pl/graphs:copy_database -- \
        --input_db="file:///users/zfisches/cc1.mysql?ml4pl_${dataset}?charset=utf8" \
        --output_db="sqlite:////$EXPORT/${dataset}_$TIMESTAMP.db" \
        --max_rows=4096
    create_tarballs
  fi
}

main() {
  # export_dataset bytecode
  export_dataset devmap_amd
  export_dataset devmap_nvidia
  export_dataset reachability
  export_dataset domtree
  export_dataset datadep
  export_dataset liveness
  export_dataset subexpressions
}
main $@
