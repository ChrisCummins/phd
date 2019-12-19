# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Print run commands for TRAINING the node lstm baselines to stdout."""


def lstm_node_series_base_script():
  # NODE LSTM SERIES BASE SCRIPT

  devices = [0, 1, 2, 3, 0, 1]
  datasets = [
    "ml4pl_reachability",
    "ml4pl_datadep",
    "ml4pl_domtree",
    "ml4pl_liveness",
    "ml4pl_polyhedra",
    "ml4pl_alias_set",
  ]
  log_db = "logs_node_lstm_series.db"

  base = "cd phd; export CUDA_VISIBLE_DEVICES={}; \
bazel run deeplearning/ml4pl/models/lstm:lstm_node_classifier -- \
--graph_db='file:///users/zfisches/cc1.mysql?{}?charset=utf8' \
--log_db='sqlite:////users/zfisches/{}' \
--bytecode_db='file:///users/zfisches/cc1.mysql?ml4pl_bytecode?charset=utf8' \
--vmodule='*'=3 \
--max_encoded_length=100000 \
--max_nodes_in_graph=25000 \
--working_dir=/users/zfisches/logs_node_lstm_20191118 \
--unlabelled_graph_db='file:///users/zfisches/cc1.mysql?ml4pl_unlabelled_corpus?charset=utf8' \
--encoded_bytecode_db='file:///users/zfisches/cc1.mysql?ml4pl_encoded_statements?charset=utf8' \
--batch_size=32 \
--bytecode_encoder=llvm \
--batch_scores_averaging_method=binary \
--max_train_per_epoch=10000 \
--max_val_per_epoch=2000 \
--notest_on_improvement \
--epoch_count=50 \
    "
  for device, dataset in zip(devices, datasets):
    if dataset == "ml4pl_liveness":
      print(base.format(device, dataset, log_db) + " --group_by=identifier")
    else:
      print(base.format(device, dataset, log_db))
    print("\n\n\n\n")


if __name__ == "__main__":
  lstm_node_series_base_script()
