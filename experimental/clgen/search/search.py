#!/usr/bin/env python3
#
# Usage:
#   ./search.py ~/src/clgen/model.json ~/src/clgen/tests/data/cl/sample-1.gs \
#       ~/src/clgen/tests/data/tiny/corpus/3.cl || less search.log
#
from random import randint
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile

import clgen
import numpy as np
from clgen import log as clgen_log
from clgen import model
from clgen import preprocess
from labm8.time import nowstr

from labm8 import fs
from phd import labm8

from io import StringIO


def features_from_file(path):
  """
  Fetch features from file.

  Arguments:
      path (str): Path to file.

  Returns:
      np.array: Feature values.
  """
  # hacky call to opencl_kernel_features and parse output
  cmd = ['opencl_kernel_features', path]
  proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
  cout, _ = proc.communicate()
  features = [
      float(x) for x in cout.decode('utf-8').split('\n')[1].split(',')[2:]
  ]
  return np.array(features)


def get_features(code):
  """
  Get features for code.

  Arguments:
      code (str): Source code.

  Returns:
      np.array: Feature values.
  """
  with NamedTemporaryFile() as outfile:
    outfile.write(code.encode("utf-8"))
    outfile.seek(0)
    features = features_from_file(outfile.name)
  return features


def get_distance(x1, x2):
  """
  Return distance between two feature vectors.
  """
  return np.linalg.norm(x1 - x2)


def get_sample(m, seed_text, seed):
  """
  Generate a new sample.

  Arguments:
      m (Model): CLgen model.
      seed_text (str): Seed text.
      seed (int): Seed value.

  Returns:
      (int, str): 0 if good sample, 1 if bad, 2 if ugly.
  """
  try:
    buf = StringIO()
    m.sample(
        seed_text=seed_text, output=buf, seed=seed, max_length=5000, quiet=True)
    out = buf.getvalue()
    result = preprocess.preprocess(out)
    return 0, result
  except preprocess.BadCodeException:
    return 1, None
  except preprocess.UglyCodeException:
    return 2, None


def get_start_code(m):
  while True:
    try:
      buf = StringIO()
      m.sample(
          seed_text='__kernel void A(__global float* a, '
          '__global float* b, __global float* c, '
          'const int d) {',
          output=buf,
          max_length=5000,
          quiet=True)
      out = buf.getvalue()
      return preprocess.preprocess(out)
    except preprocess.BadCodeException:
      pass
    except preprocess.UglyCodeException:
      pass


def get_mutation(m, start_code):
  """
  Fetch a mutation of code.

  Arguments:
      m (Model): CLgen model.
      start_code: Source code to mutate.

  Returns:
      (str, int, int, list of tuples): Mutates source code, index of start
          text, seed value, list of attempts.
  """
  min_mutate_idx = len('__kernel void ')
  max_mutate_idx = len(start_code) - 1
  attempts = ''

  max_attempts = 500

  for i in range(1, max_attempts + 1):
    # pick a random split and seed
    mutate_idx = randint(min_mutate_idx, max_mutate_idx)
    mutate_seed = randint(0, 255)

    start_text = start_code[:mutate_idx]

    print(
        ">>> attempt", i, "idx", mutate_idx, "seed", mutate_seed, "-", end=" ")
    ret, code = get_sample(m, start_text, mutate_seed)
    if not ret and code != start_code:
      print("good")
      return code, mutate_idx, mutate_seed, attempts
    else:
      if ret == 0:
        print("unmodified")
      elif ret == 1:
        print("bad")
      else:
        print("ugly")
      attempts += str(ret)
  return None, None, None, attempts


def write_log(log, logpath):
  clgen.write_file(logpath, clgen.format_json(log))


def add_to_log(log, entry, name=None):
  if name:
    log.append({"date": nowstr(), "name": name, "data": entry})
  else:
    log.append({"date": nowstr(), "data": entry})


def escape_features(features):
  return ', '.join([str(x) for x in features])


def get_entries(log, name):
  entries = []
  for entry in log:
    if entry.get("name") == name:
      entries.append(entry)
  return entries


def get_steps(log):
  return get_entries(log, "step")


def get_code_history(log):
  code_history = [log[0]['data']['start_code']]
  for step in get_steps(log):
    if step['data']['code'] != code_history[-1]:
      code_history.append(step['data']['code'])
  return code_history


def search(m, target_code, logpath, start_code=None):
  # resume search
  if fs.exists(logpath):
    log = clgen.load_json_file(logpath)
    print("resuming search of", len(get_steps(log)), "steps")
  else:
    log = []

  steps = get_steps(log)

  if start_code and not len(steps):
    code = start_code
  elif len(steps):
    code = steps[-1]['data']['code']
  else:
    code = get_start_code(m)

  target_features = get_features(target_code)
  features = get_features(code)
  distance = get_distance(target_features, features)

  if get_entries(log, "init"):
    init = get_entries(log, "init")[0]
    assert (init['data']['target_code'] == target_code)
    assert (init['data']['target_features'] == escape_features(target_features))

    # load history from log
    code_history = get_code_history(log)
  else:
    # create init entry
    add_to_log(
        log, {
            "start_code": code,
            "start_features": escape_features(features),
            "target_features": escape_features(target_features),
            "target_code": target_code,
            "distance": distance,
            "model": m.meta
        },
        name="init")
    write_log(log, logpath)
    code_history = [code]

  # keep track of best
  if len(steps):
    best = steps[-1]['data']['best']
  else:
    best = {"distance": distance, "code": code, "improvement_count": 0}

  # maximum number of mutations before stopping search
  MAX_STEPS = 1000

  for i in range(len(steps), MAX_STEPS):
    print("step", i, "of", MAX_STEPS)
    newcode, mutate_idx, mutate_seed, attempts = get_mutation(m, code)
    try:
      features = get_features(newcode)
      distance = get_distance(target_features, features)
    except ValueError:
      newcode = None

    entry = {"count": i, "attempts": attempts}

    if newcode:
      entry["base_code"] = code
      entry["code"] = newcode
      entry["distance"] = distance
      entry["distance_diff"] = 1 - distance / best["distance"]
      entry["features"] = escape_features(features)
      entry["mutate_idx"] = mutate_idx
      entry["mutate_seed"] = mutate_seed
      code_history.append(code)
    else:
      print("    -> step back")
      # step back
      if len(code_history):
        code = code_history.pop()
      entry["step_back"] = code

    if distance < best["distance"]:
      print("    -> improvement {:.1f}%".format(entry["distance_diff"] * 100))
      best["distance"] = distance
      best["code"] = newcode
      best["features"] = escape_features(features)
      best["improvement_count"] += 1
    else:
      if newcode:
        print("    -> regression {:.1f}%".format(entry["distance_diff"] * 100))

    entry["best"] = best

    add_to_log(log, entry, name="step")
    write_log(log, logpath)

    # doesn't have to be exactly zero but whatever
    if distance <= 0.001:
      print("found exact match!")
      break

  add_to_log(
      log, {
          "best_code": best['code'],
          "best_features": escape_features(best['features']),
          "best_distance": best['distance']
      },
      name="end")
  write_log(log, logpath)


def main():
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument("model", help="Path to model")
  parser.add_argument("target", help="Path to target code")
  parser.add_argument(
      "-i",
      "--input",
      metavar="path",
      default=None,
      help="Path to starting code")
  parser.add_argument(
      "-l",
      "--log",
      metavar="path",
      default="search-log.json",
      help="Path to log file")
  args = parser.parse_args()

  clgen_log.init(verbose=True)

  # load and train model
  modelpath = args.model
  if modelpath.endswith(".tar.bz2"):
    m = model.from_tar(modelpath)
  else:
    model_json = clgen.load_json_file(modelpath)
    m = clgen.model.from_json(model_json)
  m.train()

  # read target code
  with open(args.target) as infile:
    target_code = infile.read()

  # read start code if provided
  start_code = None
  if args.input:
    with open(args.input) as infile:
      start_code = infile.read()

  search(m, target_code, args.log, start_code=start_code)


if __name__ == "__main__":
  main()
