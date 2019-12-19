import hashlib
import itertools
from pathlib import Path


template_header = """#!/bin/bash
#SBATCH --job-name=T5LM{idx:03d}
#SBATCH --time={timelimit}
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ault07,ault08

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

"""

build_helper = """
# Make sure you run the build command before launching:
# {build_command}
# Alternatively run manually with
# cd phd; export CUDA_VISIBLE_DEVICES=0;
# bazel run //deeplearning/ml4pl/scripts:devmap -- 

"""

template_command = """
srun ./bazel-bin/deeplearning/ml4pl/scripts/devmap \
    {database} \
    --log_db='{log_db}' \
    --bytecode_db='file:///users/zfisches/cc1.mysql?ml4pl_bytecode?charset=utf8' \
    --working_dir='/users/zfisches/logs_table5_lstm' \
    --model=lstm \
    --batch_size=25 \
    --bytecode_encoder={encoder} \
    --max_encoded_length={max_len} \
    --groups={groups}

"""

build_command = "bazel build //deeplearning/ml4pl/scripts:devmap"

groups_list = [
  "0,1",
  "2,3",
  "4,5",
  "6,7",
  "8,9",
]

databases = [
  "--amd_graph_db='sqlite:////users/zfisches/db/devmap_amd_unbalanced_split_20191120.db'",
  "--nvidia_graph_db='sqlite:////users/zfisches/db/devmap_nvidia_unbalanced_split_20191120.db'",
]

encoder = [
  "llvm",
  "opencl",
  "inst2vec",
]

max_lens = ["25000", "24795", "25000"]


timelimit = "04:00:00"
log_db = "sqlite:////users/zfisches/logs_table5_lstm.db"
base_path = Path(
  "/users/zfisches/phd/deeplearning/ml4pl/scripts/table5_lstm/kfold/"
)


def generate():
  tmpl = template_header + template_command
  modes = zip(encoder, max_lens)
  dimensions = [modes, groups_list, databases]
  configs = list(itertools.product(*dimensions))

  base_path.mkdir(parents=True, exist_ok=True)
  readme = open(base_path / "README.txt", "w")
  print(build_helper.format(build_command=build_command), file=readme)

  for i, c in enumerate(configs):
    mode, groups, database = c
    fmtdic = {
      "timelimit": timelimit,
      "idx": i,
      "log_db": log_db,
      "database": database,
      "groups": groups,
      "encoder": mode[0],
      "max_len": mode[1],
    }

    print(f"{i:03d}: {fmtdic}", file=readme)

    info = f"\n# Run No. {i:03d}:\n# {fmtdic}\n\n"

    fmtdic.update({"build_command": build_command})
    jobstr = (
      template_header.format(**fmtdic)
      + info
      + template_command.format(**fmtdic)
      + "\n"
    )

    data = (
      "amd_unbalanced"
      if "amd_graph" in fmtdic["database"]
      else "nvidia_unbalanced"
    )
    filename = (
      f"run_{i:03d}_{data}_{fmtdic['encoder']}_{groups.replace(',','_')}.sh"
    )
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
