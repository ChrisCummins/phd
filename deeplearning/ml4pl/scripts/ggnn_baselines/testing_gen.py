# Copyright 2019-2020 the ProGraML authors.
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
import hashlib
import itertools
from pathlib import Path


template_header = """#!/bin/bash
#SBATCH --job-name=BlsT{idx:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08
#SBATCH --cpus-per-task=8

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

"""

build_helper = """
# Make sure you run the build command before launching:
# {build_command}
# Alternatively run manually with
# cd phd; export CUDA_VISIBLE_DEVICES=0;
# bazel run //deeplearning/ml4pl/models/ggnn -- 

"""

build_command = "bazel build //deeplearning/ml4pl/models/ggnn"

template_command = """# TEST-RUN COMMAND
srun ./bazel-bin/deeplearning/ml4pl/models/ggnn \
    --graph_db='file:///users/zfisches/cc1.mysql?{database}?charset=utf8' \
    --log_db='{log_db}' \
    --working_dir='/users/zfisches/logs_20191117' \
    --ggnn_layer_timesteps=30 \
    --epoch_count=120 \
    --batch_scores_averaging_method=binary \
    --max_train_per_epoch=10000 \
    --max_val_per_epoch=2000 \
    --vmodule='*'=5 \
    --batch_size=800000 \
    --inst2vec_embeddings=random \
    --notest_on_improvement \
    --position_embeddings=fancy \
    --learning_rate=0.0001 \
    --use_lr_schedule \
    --restore_model={restore} \
    --test_only \
    --graph_reader_buffer_size=4096 \
    --ggnn_unroll_strategy={ggnn_unroll_strategy} \
    --ggnn_unroll_factor={ggnn_unroll_factor}

"""

strategies = [
  "none",
  "data_flow_max_steps",
  "label_convergence",
]  # later add 'constant', 'edge_count'
factors = ["0", "0", "0"]  # for constant etc. we need more

timelimit = "04:00:00"
log_db = "sqlite:////users/zfisches/ggnn_case_study_logs.db"
base_path = Path(
  "/users/zfisches/phd/deeplearning/ml4pl/scripts/ggnn_baselines/testing/"
)

restore_from = [
  "20191117T024211@ault05.cscs.ch:102",
  "20191117T031908@ault09.cscs.ch:89",
  "20191117T032950@ault06.cscs.ch:47",
  "20191117T033531@ault06.cscs.ch:52",
  "20191117T160503@ault09.cscs.ch:82",
]

databases = [
  "ml4pl_reachability",
  "ml4pl_datadep",
  "ml4pl_domtree",
  "ml4pl_subexpressions",
  "ml4pl_liveness",
]


def generate():
  tmpl = template_header + template_command
  modes = zip(strategies, factors)
  models = zip(databases, restore_from)
  dimensions = [modes, models]
  configs = list(itertools.product(*dimensions))

  base_path.mkdir(parents=True, exist_ok=True)
  readme = open(base_path / "README.txt", "w")
  print(build_helper.format(build_command=build_command), file=readme)

  for i, c in enumerate(configs):
    mode, model = c
    fmtdic = {
      "timelimit": timelimit,
      "idx": i,
      "log_db": log_db,
      "restore": model[1],
      "database": model[0],
      "ggnn_unroll_strategy": mode[0],
      "ggnn_unroll_factor": mode[1],
    }

    print(f"{i:03d}: {fmtdic}", file=readme)

    info = f"\n# Run No. {i:03d}:\n# {fmtdic}\n\n"
    jobstr = (
      template_header.format(**fmtdic)
      + info
      + template_command.format(**fmtdic)
      + "\n"
    )

    filename = f"run_{i:03d}_{fmtdic['database']}_{fmtdic['ggnn_unroll_strategy']}_{fmtdic['ggnn_unroll_factor']}.sh"
    path = base_path / filename
    write_file(path, jobstr)
  readme.close()
  print("Success.")
  print(build_helper.format(build_command=build_command))
  print(build_command)


def stamp(stuff):
  hash_object = hashlib.sha1(str(stuff).encode("utf-8"))
  hex_dig = hash_object.hexdigest()
  return hex_dig[:7]


def write_file(path: Path, content: str):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    f.write(content)


if __name__ == "__main__":
  generate()
