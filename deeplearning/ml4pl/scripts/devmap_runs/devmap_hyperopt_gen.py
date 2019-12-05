import hashlib
import itertools
import os
from pathlib import Path


def stamp(stuff):
  hash_object = hashlib.sha1(str(stuff).encode("utf-8"))
  hex_dig = hash_object.hexdigest()
  return hex_dig[:7]


def ggnn_devmap_hyperopt(
  start_step=0, gpus=[0, 1, 2, 3], how_many=None, test_splits="kfold"
):
  # GGNN DEVMAP HYPER OPT SERIES
  # fix
  log_db = "ggnn_devmap_hyperopt.db"
  # devices = [0,1,2,3]
  # flexible
  state_drops = ["1.0", "0.95", "0.9"]
  timestep_choices = ["2,2,2", "3,3", "30"]
  datasets = ["amd", "nvidia"]
  batch_sizes = ["18000", "40000", "9000"]
  out_drops = ["0.5", "0.8", "1.0"]
  edge_drops = ["0.8", "1.0", "0.9"]
  embs = ["constant", "random"]
  pos_choices = ["off", "fancy"]

  # order is important!
  hyperparam_keys = [
    "state_drop",
    "timesteps",
    "dataset",
    "batch_size",
    "out_drop",
    "edge_drop",
    "emb",
    "pos",
  ]
  opt_space = [
    [state_drops[0]],
    [timestep_choices[0]],
    datasets,
    [batch_sizes[0]],
    out_drops,
    edge_drops,
    embs,
    pos_choices,
  ]
  configs = list(itertools.product(*opt_space))
  length = len(configs)
  print(f"Generating {length} configs per test group.\n")

  # None -> all.
  if not how_many:
    how_many = length - start_step

  # cd phd; export CUDA_VISIBLE_DEVICES={device}; \

  template = """#!/bin/bash
#SBATCH --job-name=dvmp{i:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

# Make sure you run the build command before launching:
# {build_command}
# Alternatively run manually with
# cd phd; export CUDA_VISIBLE_DEVICES=0;
# bazel run //deeplearning/ml4pl/models/ggnn:ggnn --

srun ./bazel-bin/deeplearning/ml4pl/models/ggnn \
--graph_db='sqlite:////users/zfisches/db/devmap_{dataset}_20191113.db' \
--log_db='sqlite:////users/zfisches/{log_db}' \
--working_dir='/users/zfisches/logs_ggnn_devmap_20191117' \
--epoch_count=150 \
--alsologtostderr \
--position_embeddings={pos} \
--ggnn_layer_timesteps={timesteps} \
--inst2vec_embeddings={emb} \
--output_layer_dropout_keep_prob={out_drop} \
--graph_state_dropout_keep_prob={state_drop} \
--edge_weight_dropout_keep_prob={edge_drop} \
--batch_size={batch_size} \
--manual_tag=HyperOpt-{i:03d}-{stamp} \
"""  # NO WHITESPACE!!!

  build_command = "bazel build //deeplearning/ml4pl/models/ggnn"

  if test_splits == "kfold":
    template += " --kfold"
    test_splits = ["kfold"]
  else:
    template += " --test_split={test_split} --val_split={val_split}"
  for g in test_splits:
    path = base_path / str(g)
    path.mkdir(parents=True, exist_ok=True)
    readme = open(path / "README.txt", "w")
    print(f"############### TEST GROUP {g} ##############\n", file=readme)
    print(hyperparam_keys, file=readme)
    print("\n", file=readme)
    for i in range(start_step, start_step + how_many):
      config = dict(zip(hyperparam_keys, configs[i]))
      stmp = stamp(config)
      print(f"HyperOpt-{i:03d}-{stmp}: " + str(config), file=readme)
      config.update(
        {
          "stamp": stmp,
          "log_db": log_db,
          "i": i,
          "build_command": build_command,
          # 'device': gpus[i % len(gpus)]
        }
      )
      if not g == "kfold":
        config.update(
          {"timelimit": "00:30:00", "test_split": g, "val_split": (g + 1) % 9}
        )
      else:  # kfold 4h
        config.update({"timelimit": "04:00:00"})

      print(template.format(**config))
      print("\n")

      with open(base_path / str(g) / f"run_{g}_{i:03d}_{stmp}.sh", "w") as f:
        f.write(template.format(**config))
        f.write(f"\n# HyperOpt-{i:03d}-{stmp}:")
        f.write(f"\n# {config}\n")
    readme.close()
  readme.close()
  print("Success.")
  # print(build_helper.format(build_command=build_command))
  print("Build before run with:\n")
  print(build_command)


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1:
    tg = sys.argv[1]
    tgs = [int(tg)]
  else:
    tgs = "kfold"
  base_path = Path(
    "/users/zfisches/phd/deeplearning/ml4pl/scripts/devmap_runs/"
  )
  ggnn_devmap_hyperopt(test_splits=tgs)
