#!/usr/bin/env python3
#
# Searching the program space.
#
import os
from random import randint
from subprocess import PIPE
from subprocess import Popen
from sys import exit
from tempfile import NamedTemporaryFile

import numpy as np
from smith import preprocess


def sample(seed, mutate_idx=-1, mutate_seed=0, start="start.txt"):
  os.chdir(os.path.expanduser("~/src/torch-rnn"))
  cmd = [
    "th",
    "sample.lua",
    "-temperature",
    "0.9",
    "-checkpoint",
    "cv/checkpoint_530000.t7",
    "-length",
    "11000",
    "-opencl",
    "1",
    "-n",
    "1",
    "-start",
    start,
    "-seed",
    str(seed),
    "-mutate_idx",
    str(mutate_idx),
    "-mutate_seed",
    str(mutate_seed),
  ]
  proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
  cout, _ = proc.communicate()
  return cout.decode("utf-8")


def write_file(path, data):
  with open(path, "w") as outfile:
    print(data, file=outfile)


def indent(data):
  return "\n".join(["  " + str(l or "") for l in str(data).split("\n")])


def generate_test_code(start_seed=0):
  seed = start_seed - 1
  found_sample = False

  while True:
    seed += 1
    print("Sampling seed {} ... ".format(seed), end="")
    code = sample(seed)
    result = None
    try:
      result = preprocess.preprocess(code)
    except Exception:
      print("bad")
      print(indent(code[:300]))
    if result and len(code) > 100:
      path = os.path.join("search", str(seed) + ".txt")
      write_file(path, result)
      print(path)
      print(indent(result))


_LOGPATH = None


def init_log(logpath):
  global _LOGPATH
  _LOGPATH = logpath
  with open(logpath, "w") as outfile:
    print(end="", file=outfile)


def log(*args, **kwargs):
  print(*args, **kwargs)
  if _LOGPATH:
    with open(_LOGPATH, "a") as outfile:
      kwargs["file"] = outfile
      print(*args, **kwargs)


def get_features(code):
  with NamedTemporaryFile() as outfile:
    outfile.write(code.encode("utf-8"))
    outfile.seek(0)
    features = FeaturesFromFile(outfile.name)
  return features


def FeaturesFromFile(path):
  # Hacky call to smith-features and parse output
  cmd = ["smith-features", path]
  proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
  cout, _ = proc.communicate()
  features = [
    float(x) for x in cout.decode("utf-8").split("\n")[1].split(",")[2:]
  ]
  return np.array(features)


def get_distance(x1, x2):
  return np.linalg.norm(x1 - x2)


def get_sample(start_txt, seed):
  with open("start.txt", "w") as outfile:
    outfile.write(start_txt)

  try:
    s = sample(seed)
    result = preprocess.preprocess(s)
    if result:
      return result
  except Exception as e:
    pass


def get_mutation(input):
  min_mutate_idx = len("__kernel void ")
  max_mutate_idx = len(input) - 1

  while True:
    # Pick a random split and seed
    mutate_idx = randint(min_mutate_idx, max_mutate_idx)
    mutate_seed = randint(0, 255)

    start_text = input[:mutate_idx]
    log(">>> trying mutate idx", mutate_idx, "seed", mutate_seed)

    code = get_sample(start_text, mutate_seed)
    if code:
      return code, mutate_idx, mutate_seed


def search(input, benchmark, datadir, logpath="search.log"):
  init_log(os.path.join(datadir, logpath))
  target_features = get_features(benchmark)

  code = input
  features = get_features(code)
  distance = get_distance(target_features, features)

  log(">>> starting code:" + "\n" + indent(code))
  log(">>> target code:" + "\n" + indent(benchmark))

  log(">>> target features:  ", ", ".join([str(x) for x in target_features]))
  log(
    ">>> starting features:",
    ", ".join([str(x) for x in features]),
    "distance",
    distance,
  )

  best = {"distance": distance, "idx": -1, "seed": 0, "code": code}

  count = 0
  improved_counter = 0
  while True:
    ret = get_mutation(code)
    newcode, mutate_idx, mutate_seed = ret
    features = get_features(newcode)
    distance = get_distance(target_features, features)

    count += 1

    log(
      ">>>",
      count,
      "new mutation",
      "features:",
      ", ".join([str(x) for x in features]),
      "distance:",
      distance,
      "( best:",
      best["distance"],
      ")",
    )
    log(indent(newcode))

    if distance < best["distance"]:
      improved_counter += 1
      log(
        "-> best feature distance reduced from",
        best["distance"],
        "to",
        distance,
        "(-{:.2f}%)".format(((1 - distance / best["distance"]) * 100)),
        "improved_counter:",
        improved_counter,
      )
      best["distance"] = distance
      best["idx"] = mutate_idx
      best["seed"] = mutate_seed
      best["code"] = newcode

      outpath = os.path.join(
        datadir, "search-step-{}.cl".format(improved_counter)
      )
      with open(outpath, "w") as outfile:
        print("/* Iteration #{} */".format(count), file=outfile)
        print("/* Improvement #{} */".format(improved_counter), file=outfile)
        print(
          "/* Target features: {} */".format(
            ", ".join([str(x) for x in target_features])
          ),
          file=outfile,
        )
        print(
          "/*        Features: {} */".format(
            ", ".join([str(x) for x in features])
          ),
          file=outfile,
        )
        print("/* Distance: {} */".format(distance), file=outfile)
        print(newcode, file=outfile)

      # Doesn't have to be exactly zero but whatever.
      if distance <= 0.000001:
        log(">>> found exact match")
        exit(0)


def main():
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("benchmark")
  parser.add_argument("datadir")
  args = parser.parse_args()

  with open(args.input) as infile:
    input = infile.read()

  with open(args.benchmark) as infile:
    benchmark = infile.read()

  search(input, benchmark, args.datadir)


if __name__ == "__main__":
  main()
