# preamble

import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from clgen.atomizer import CharacterAtomizer
from keras.preprocessing.sequence import pad_sequences
from scipy.stats.mstats import gmean
from sklearn.model_selection import StratifiedKFold

from labm8 import fs

# plotting config:
sns.set(style="ticks", color_codes=True)
plt.style.use(["seaborn-white", "seaborn-paper"])

# methods for data wrangling:


def escape_suite_name(g):
  """format benchmark suite name for display"""
  c = g.split('-')
  if (c[0] == "amd" or c[0] == "npb" or c[0] == "nvidia" or c[0] == "shoc"):
    return c[0].upper()
  elif (c[0] == "parboil" or c[0] == "polybench" or c[0] == "rodinia"):
    return c[0].capitalize()
  else:
    return "CLgen"


def escape_benchmark_name(g):
  """escape benchmark name for display"""
  c = g.split('-')
  suite = escape_suite_name(c[0])
  if suite == "CLgen":
    return suite
  else:
    return suite + "." + c[-2]


def get_2(D):
  """ return np array of shape (len(D), nb_features)"""
  return np.array([
      D["transfer"].values,
      D["wgsize"].values,
  ]).T


def get_4(D):
  """ return np array of shape (len(D), nb_features)"""
  return np.array([
      (D["transfer"].values / (D["comp"].values + D["mem"].values)),  # F1
      (D["coalesced"].values / D["mem"].values),  # F2
      ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),  # F3
      (D["comp"].values / D["mem"].values),  # F4
  ]).T


def get_11(D):
  """ return np array of shape (len(D), nb_features)"""
  return np.array([
      D["comp"].values,
      D["rational"].values,
      D["mem"].values,
      D["localmem"].values,
      D["coalesced"].values,
      D["transfer"].values,
      D["wgsize"].values,
      (D["transfer"].values / (D["comp"].values + D["mem"].values)),  # F1
      (D["coalesced"].values / D["mem"].values),  # F2
      ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),  # F3
      (D["comp"].values / D["mem"].values),  # F4
  ]).T


def get_sequences(D, max_seq_len):
  """ return np array of shape (len(D), max_seq_len) """
  for row in D["seq"].values:
    assert (len(row) == max_seq_len)
  data = np.array(D["seq"].values)
  return np.vstack([np.expand_dims(x, axis=0) for x in data])


def get_labels(D):
  """ cpu/gpu to int, return np array of shape (len(D), 1) """
  return np.vstack([np.expand_dims(x, axis=0) for x in D["oracle_enc"]])


def get_y2(D):
  l2 = [x[0] for x in get_labels(D)]
  l1 = [not x for x in l2]
  return np.array(list(zip(l1, l2)), dtype=np.int32)


def get_train_test_splits(D, n_splits=10, seed=1):
  synthetics = D[D["synthetic"] == 1]
  benchmarks = D[D["synthetic"] == 0]
  assert (len(synthetics) + len(benchmarks) == len(D))

  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  T = []
  for train_index, test_index in skf.split(benchmarks.index,
                                           benchmarks["oracle_enc"]):
    indices = list(train_index) + list(test_index)
    assert len(indices) == len(benchmarks)
    assert set(benchmarks.index) == set(indices)

    T.append((pd.concat((synthetics, benchmarks.loc[train_index])),
              benchmarks.loc[test_index]))
  return T


def load_data_desc(platform,
                   source="B",
                   max_seq_len=1000,
                   atomizer=CharacterAtomizer,
                   quiet=False):
  """ load experimental results """

  def get_benchmarks(platform):
    B = pd.read_csv(
        fs.path("runtimes/{platform}-benchmarks.csv".format(**vars())))
    B["source"] = [escape_suite_name(x) for x in B["benchmark"]]
    B["synthetic"] = [0] * len(B)
    return B

  def get_npb_benchmarks(platform):
    B = get_benchmarks(platform)
    msk = B["source"] == "NPB"
    return B[msk]

  def get_synthetics(platform):
    S = pd.read_csv(fs.path("runtimes/{platform}-clgen.csv".format(**vars())))
    S["source"] = ["CLgen"] * len(S)
    S["synthetic"] = [1] * len(S)
    return S

  if source == "B":
    dataframe = get_benchmarks(platform)
  elif source == "S":
    dataframe = get_synthetics(platform)
  elif source == "BS":
    dataframe = pd.concat((get_benchmarks(platform), get_synthetics(platform)))
  elif source == "N":
    dataframe = get_npb_benchmarks(platform)
  elif source == "NS":
    dataframe = pd.concat((get_npb_benchmarks(platform),
                           get_synthetics(platform)))
  else:
    raise Exception

  dataframe["oracle_enc"] = [
      1 if x == "GPU" else 0 for x in dataframe["oracle"].values
  ]
  dataframe["benchmark_name"] = [
      escape_benchmark_name(b) for b in dataframe["benchmark"].values
  ]

  # load source code:
  source_dir = fs.path("kernels")
  srcs, benchmark_names = [], []
  for row in dataframe["benchmark"].values:
    inpath = fs.path(source_dir, row + ".cl")
    with open(inpath) as infile:
      src = infile.read()
    if not src.startswith("__kernel void A"):
      print(fs.basename(inpath))
      raise Exception(src)
    srcs.append(src)
  dataframe["src"] = srcs
  dataframe["src_len"] = [len(s) for s in srcs]

  if not quiet:
    print("num instances {} ({} synthetic, {} benchmarks)".format(
        len(dataframe), sum(dataframe["synthetic"].values),
        len(dataframe) - sum(dataframe["synthetic"].values)))
    print("unique kernels", len(set(srcs)))

  # encode and pad sequences:
  atomizer = atomizer.from_text(''.join(dataframe["src"].values))

  seqs = [atomizer.atomize(seq) for seq in dataframe["src"].values]
  seq_length = min(max(len(s) for s in seqs), max_seq_len)
  pad_val = atomizer.vocab_size + 1
  dataframe["seq_len"] = [len(s) for s in seqs]
  dataframe["seq"] = list(pad_sequences(seqs, maxlen=seq_length, value=pad_val))

  if not quiet:
    print("vocab size", atomizer.vocab_size + 1)
    print("pad val", pad_val)
    print("padded seq length", seq_length)

  return {
      "dataframe": dataframe,
      "seq_length": seq_length,
      "atomizer": atomizer
  }


def get_training_data(data_desc,
                      *args,
                      seed=204,
                      n_splits=10,
                      split_i=0,
                      **kwargs):
  dataframe = data_desc["dataframe"]
  seq_length = data_desc["seq_length"]
  train, test = get_train_test_splits(
      dataframe, seed=seed, n_splits=n_splits)[split_i]

  x_train_2 = get_2(train)
  x_train_4 = get_4(train)
  x_train_11 = get_11(train)
  x_train_seq = get_sequences(train, seq_length)
  y_train = get_labels(train)
  y2_train = get_y2(train)

  x_test_2 = get_2(test)
  x_test_4 = get_4(test)
  x_test_11 = get_11(test)
  x_test_seq = get_sequences(test, seq_length)
  y_test = get_labels(test)
  y2_test = get_y2(test)

  return (
      {
          "dataframe": train,
          "x_2": x_train_2,
          "x_4": x_train_4,
          "x_11": x_train_11,
          "x_seq": x_train_seq,
          "y": y_train,
          "y_2": y2_train
      },
      {
          "dataframe": test,
          "x_2": x_test_2,
          "x_4": x_test_4,
          "x_11": x_test_11,
          "x_seq": x_test_seq,
          "y": y_test,
          "y_2": y2_test
      },
  )


# analyze results:


def enc2key(p):
  return "runtime_gpu" if p else "runtime_cpu"


def analyze(predictions, test):
  frame = test["dataframe"]
  oracle = np.array(frame["oracle_enc"], dtype=np.bool)
  incorrect = np.logical_xor(predictions, oracle)
  correct = np.logical_not(incorrect)

  zero_r = Counter(oracle).most_common(1)[0][0]
  zero_r_key = enc2key(zero_r)

  speedups = np.array([
      min(d["runtime_cpu"], d["runtime_gpu"]) / d[enc2key(p)]
      for p, d in zip(predictions,
                      frame.T.to_dict().values())
  ])
  speedup_avg = speedups.mean()
  speedup_geo = gmean(speedups)

  accuracy = sum(correct) / len(test["dataframe"])

  confusion_matrix = np.zeros((2, 2), dtype="int32")
  confusion_matrix[0][0] = sum(
      np.logical_and(np.logical_not(predictions), np.logical_not(oracle)))
  confusion_matrix[0][1] = sum(
      np.logical_and(predictions, np.logical_not(oracle)))
  confusion_matrix[1][0] = sum(
      np.logical_and(np.logical_not(predictions), oracle))
  confusion_matrix[1][1] = sum(np.logical_and(predictions, oracle))

  assert (confusion_matrix.sum() == len(test["dataframe"]))
  assert (confusion_matrix[0][1] + confusion_matrix[1][1] == sum(predictions))
  assert (confusion_matrix[0][1] + confusion_matrix[1][0] == sum(incorrect))
  assert (confusion_matrix[0][0] + confusion_matrix[1][1] == sum(correct))
  print(confusion_matrix)

  return {
      "accuracy": accuracy,
      "correct": correct,
      "confusion_matrix": confusion_matrix,
      "speedups": speedups,
      "speedup_min": min(speedups),
      "speedup_max": max(speedups),
      "speedup_avg": speedup_avg,
      "speedup_geo": speedup_geo,
  }


def _nop(*args, **kwargs):
  return {}


### UTILITY
#
# def get_model_path(model_desc, platform, source,
#                    atomizer="CharacterAtomizer", maxlen=1024, seed=204):
#     split_txt = ":".join("{:02d}".format(round(d * 100)) for d in split)
#     name = model_desc["name"]
#     return "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.model".format(**vars())


def train_and_save(model_desc,
                   platform,
                   source,
                   atomizer="CharacterAtomizer",
                   maxlen=1024,
                   n_splits=10,
                   split_i=0,
                   seed=204):
  np.random.seed(seed)

  name = model_desc["name"]
  outpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.model".format(
      **vars())
  if not fs.exists(outpath):
    create_fn = model_desc.get("create_model", _nop)
    train_fn = model_desc.get("train_fn", _nop)
    save_fn = model_desc["save_fn"]
    _atomizer = globals().get(atomizer)

    # load training data
    data_desc = load_data_desc(
        platform=platform,
        source=source,
        max_seq_len=maxlen,
        atomizer=_atomizer)
    train, test = get_training_data(
        data_desc, seed=seed, split_i=split_i, n_splits=n_splits)

    # create model
    model = create_fn(seed=seed, data_desc=data_desc)

    # train model
    train_fn(
        model=model, train=train, seed=seed, platform=platform, source=source)

    fs.mkdir("models/{name}".format(**vars()))
    save_fn(outpath, model)
    print("model saved as", outpath)

  # evaluate model
  return load_and_test(
      model_desc,
      platform,
      source,
      n_splits=n_splits,
      split_i=split_i,
      atomizer=atomizer,
      maxlen=maxlen,
      seed=seed)


def load_and_test(model_desc,
                  platform,
                  source,
                  atomizer="CharacterAtomizer",
                  maxlen=1024,
                  n_splits=10,
                  split_i=0,
                  seed=204):
  np.random.seed(seed)

  name = model_desc["name"]
  inpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.model".format(
      **vars())
  outpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.result".format(
      **vars())

  if fs.exists(outpath):
    return load_result(
        model_desc,
        platform,
        source,
        n_splits=n_splits,
        split_i=split_i,
        atomizer=atomizer,
        maxlen=maxlen,
        seed=seed)
  if not fs.exists(inpath):
    return False

  test_fn = model_desc["test_fn"]
  load_fn = model_desc["load_fn"]

  # load training data
  _atomizer = globals().get(atomizer)
  data_desc = load_data_desc(
      platform=platform,
      source=source,
      max_seq_len=maxlen,
      atomizer=_atomizer,
      quiet=True)
  train, test = get_training_data(
      data_desc, seed=seed, split_i=split_i, n_splits=n_splits)

  # load model
  model = load_fn(inpath)
  print("model loaded from", inpath)

  # test model
  predictions = test_fn(model=model, test=test, seed=seed)
  analysis = analyze(predictions, test)
  test.update(analysis)
  test["predictions"] = predictions

  with open(outpath, 'wb') as outfile:
    pickle.dump(test, outfile)
  print("result saved to", outpath)

  return test


def benchmark_inference(model_desc,
                        platform,
                        source,
                        atomizer="CharacterAtomizer",
                        maxlen=1024,
                        n_splits=10,
                        split_i=0,
                        seed=204,
                        n_runtimes=100):
  np.random.seed(seed)

  name = model_desc["name"]
  inpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.model".format(
      **vars())
  outpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.result".format(
      **vars())

  if not fs.exists(inpath):
    return False

  test_fn = model_desc["test_fn"]
  load_fn = model_desc["load_fn"]

  # load training data
  _atomizer = globals().get(atomizer)
  data_desc = load_data_desc(
      platform=platform,
      source=source,
      max_seq_len=maxlen,
      atomizer=_atomizer,
      quiet=True)
  train, test = get_training_data(
      data_desc, seed=seed, split_i=split_i, n_splits=n_splits)

  # load model
  model = load_fn(inpath)
  print("model loaded from", inpath)

  # test model
  runtimes = []
  for i in range(n_runtimes):
    start = time.time()
    predictions = test_fn(model=model, test=test, seed=seed)
    elapsed = (time.time() - start) / len(test["y"])
    runtimes.append(elapsed)

  return np.array(runtimes)


def source2str(s):
  if s == "B":
    return "Benchmarks"
  elif s == "S":
    return "CLgen"
  elif s == "BS":
    return "w. CLgen"
  else:
    raise Exception


def platform2str(p):
  if p == "amd":
    return "AMD Tahiti 7970"
  elif p == "nvidia":
    return "NVIDIA GTX 970"
  else:
    raise Exception


def load_result(model_desc,
                platform,
                source,
                atomizer="CharacterAtomizer",
                maxlen=1024,
                n_splits=10,
                split_i=0,
                seed=204):
  name = model_desc["name"]
  inpath = "models/{name}/{platform}-{source}-{atomizer}:{maxlen}-{seed}-{n_splits}-{split_i}.result".format(
      **vars())
  if not fs.exists(inpath):
    return False

  with open(inpath, 'rb') as infile:
    result = pickle.load(infile)

  return result
