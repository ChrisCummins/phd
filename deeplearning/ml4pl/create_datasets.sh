#!/usr/bin/env bash
# A script to create the labelled graph datasets.

set -eux

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_devmap_amd" || true
bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
    --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_devmap?charset=utf8' \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_devmap_amd?charset=utf8" \
    --gpu='amd_tahiti_7970' \
    --alsologtostderr

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_devmap_nvidia" || true
bazel run //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset -- \
    --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_devmap?charset=utf8' \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_devmap_nvidia?charset=utf8" \
    --gpu='nvidia_gtx_960' \
    --alsologtostderr

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_reachability" || true
bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
    --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus?charset=utf8' \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_reachability?charset=utf8" \
    --analysis='reachability' \
    --alsologtostderr

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_domtree" || true
bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
    --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus?charset=utf8' \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_domtree?charset=utf8" \
    --analysis="dominator_tree" \
    --alsologtostderr

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_datadep" || true
bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
    --input_db="file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus?charset=utf8" \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_datadep?charset=utf8" \
    --analysis="data_dependence" \
    --alsologtostderr

mysql -h cc1.inf.ed.ac.uk -e "drop schema ml4pl_liveness" || true
bazel run //deeplearning/ml4pl/graphs/labelled:make_data_flow_analysis_dataset -- \
    --input_db="file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus?charset=utf8" \
    --output_db="file:///var/phd/db/cc1.mysql?ml4pl_liveness?charset=utf8" \
    --analysis="liveness" \
    --alsologtostderr
