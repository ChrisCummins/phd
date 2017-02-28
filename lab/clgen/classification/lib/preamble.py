# preamble

import clgen
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import time

from clgen import clutil
from clgen.atomizer import CharacterAtomizer, GreedyAtomizer

from collections import Counter

from IPython.display import SVG

from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.visualize_util import model_to_dot
from keras.wrappers.scikit_learn import KerasClassifier

from labm8 import fs
from labm8 import viz

from scipy.stats import percentileofscore
from scipy.stats.mstats import gmean
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# methods for data wrangling:

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
        assert(len(row) == max_seq_len)
    data = np.array(D["seq"].values)
    return np.vstack([np.expand_dims(x, axis=0) for x in data])


def get_labels(D):
    """ cpu/gpu to int, return np array of shape (len(D), 1) """
    return np.vstack([np.expand_dims(x, axis=0) for x in D["oracle_enc"]])


def get_y2(D):
    l2 = [x[0] for x in get_labels(D)]
    l1 = [not x for x in l2]
    return np.array(list(zip(l1, l2)), dtype=np.int32)


def get_train_validation_test_splits(D, split=(.6, .2, .2), seed=1):
    """ split dataframe into 3 frames for training, validation, and testing """
    if round(sum(split), 3) != 1.000:  # FIXME
        print(round(sum(split), 3))
    assert(round(sum(split), 3) == 1.000)
    train_split, validation_split, test_split = split

    num_synthetic = sum(D["synthetic"].values)

    np.random.seed(seed)
    train_msk = np.random.rand(len(D)) < train_split

    train = D[train_msk]
    other = D[~train_msk]

    test_msk = np.random.rand(len(other)) < split[2] / sum(split[1:])
    test = other[test_msk]
    validation = other[~test_msk]

    return train, validation, test


def get_train_validation_test_splits(D, split=(.6, .2, .2), seed=1):
    """ split dataframe into 3 frames for training, validation, and testing.
    synthetics are exclusively for training """
    # TODO: balance benchmark suites in splits
    # TODO: balance labels in splits
    if round(sum(split), 3) != 1.000:  # FIXME
        print(round(sum(split), 3))
    assert(round(sum(split), 3) == 1.0)
    train_split, validation_split, test_split = split

    num_synthetic = sum(D["synthetic"].values)
    num_benchmarks = len(D) - num_synthetic

    np.random.seed(seed)

    if num_benchmarks:
        train_msk = np.logical_or(np.random.rand(len(D)) < train_split, D["synthetic"])
    else:
        train_msk = np.random.rand(len(D)) < train_split

    train = D[train_msk]
    other = D[~train_msk]

    test_msk = np.random.rand(len(other)) < split[2] / sum(split[1:])
    test = other[test_msk]
    validation = other[~test_msk]

    np.random.seed()  # re-seed RNG
    return train, validation, test


def load_dataframe(platform, source="B",
                   max_seq_len=1000, atomizer=CharacterAtomizer,
                   quiet=False):
    """ load experimental results """
    def escape_suite_name(g):
        """format benchmark suite name for display"""
        c = g.split('-')
        if (c[0] == "amd" or c[0] == "npb" or c[0] == "nvidia" or c[0] == "shoc"):
            return c[0].upper()
        else:
            return c[0].capitalize()

    def get_benchmarks(platform):
        B = pd.read_csv(fs.path("runtimes/{platform}-benchmarks.csv".format(**vars())))
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
        dataframe = pd.concat((get_npb_benchmarks(platform), get_synthetics(platform)))
    else:
        raise Exception

    dataframe["oracle_enc"] = [1 if x == "GPU" else 0 for x in dataframe["oracle"].values]

    # load source code:
    source_dir = fs.path("kernels")
    srcs = []
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
            len(dataframe),
            sum(dataframe["synthetic"].values),
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


def get_training_data(data_desc, seed, *args, split=(.6, .2, .2), **kwargs):
    dataframe = data_desc["dataframe"]
    seq_length = data_desc["seq_length"]
    train, validation, test = get_train_validation_test_splits(
        dataframe, seed=seed, split=split)

    x_train_2 = get_2(train)
    x_train_4 = get_4(train)
    x_train_11 = get_11(train)
    x_train_seq = get_sequences(train, seq_length)
    y_train = get_labels(train)
    y2_train = get_y2(train)

    x_val_2 = get_2(validation)
    x_val_4 = get_4(validation)
    x_val_11 = get_11(validation)
    x_val_seq = get_sequences(validation, seq_length)
    y_val = get_labels(validation)
    y2_val = get_y2(validation)

    x_test_2 = get_2(test)
    x_test_4 = get_4(test)
    x_test_11 = get_11(test)
    x_test_seq = get_sequences(test, seq_length)
    y_test = get_labels(test)
    y2_test = get_y2(test)

    return (
        {"dataframe": train, "x_2": x_train_2, "x_4": x_train_4, "x_11": x_train_11, "x_seq": x_train_seq, "y": y_train, "y_2": y2_train},
        {"dataframe": validation, "x_2": x_val_2, "x_4": x_val_4, "x_11": x_val_11, "x_seq": x_val_seq, "y": y_val, "y_2": y2_val},
        {"dataframe": test, "x_2": x_test_2, "x_4": x_test_4, "x_11": x_test_11, "x_seq": x_test_seq, "y": y_test, "y_2": y2_test},
    )

# analyze results:

def analyze(predictions, test):
    def enc2key(p):
        return "runtime_gpu" if p else "runtime_cpu"

    frame = test["dataframe"]
    oracle = np.array(frame["oracle_enc"], dtype=np.bool)
    incorrect = np.logical_xor(predictions, oracle)
    correct = np.logical_not(incorrect)

    zero_r = Counter(oracle).most_common(1)[0][0]
    zero_r_key = enc2key(zero_r)

    speedups = np.array([min(d["runtime_cpu"], d["runtime_gpu"]) / d[enc2key(p)]
                         for p,d in zip(predictions, frame.T.to_dict().values())])
    speedup_avg = speedups.mean()
    speedup_geo = gmean(speedups)

    accuracy = sum(correct) / len(test["dataframe"])

    confusion_matrix = np.zeros((2, 2), dtype="int32")
    confusion_matrix[0][0] = sum(np.logical_and(np.logical_not(predictions), np.logical_not(oracle)))
    confusion_matrix[0][1] = sum(np.logical_and(predictions, np.logical_not(oracle)))
    confusion_matrix[1][0] = sum(np.logical_and(np.logical_not(predictions), oracle))
    confusion_matrix[1][1] = sum(np.logical_and(predictions, oracle))

    assert(confusion_matrix.sum() == len(test["dataframe"]))
    assert(confusion_matrix[0][1] + confusion_matrix[1][1] == sum(predictions))
    assert(confusion_matrix[0][1] + confusion_matrix[1][0] == sum(incorrect))
    assert(confusion_matrix[0][0] + confusion_matrix[1][1] == sum(correct))
    print(confusion_matrix)

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "predictions": predictions,
        "speedups": speedups,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "speedup_avg": speedup_avg,
        "speedup_geo": speedup_geo,
    }


def _nop(*args, **kwargs):
        return {}

def evaluate(data_desc, model_desc, seed=204, split=(.6, .2, .2), n=10, quiet=False):
    """
    Model desc members:
        create_model (function): Function with definition:
                create_model(seed, i) -> model
        train_fn (function): Function with definition:
                train_fn(seed, model, train_data, validation_data) -> {}
        save_fn (function): Function with definition:
                save_fn(seed, i, model) -> None
        test_fn (function): Function with definition:
                test_fn(seed, model, test_data) -> {}

    Arguments:
        n (int): Number of repetitions

    Returns:
        list: list of length n, each element a pair of train_fn(), test_fn() return values
    """
    dataframe = data_desc["dataframe"]
    seq_length = data_desc["seq_length"]
    atomizer = data_desc["atomizer"]

    def _nop(*args, **kwargs):
        return {}

    create_model = model_desc.get("create_model", _nop)
    train_fn = model_desc.get("train_fn", _nop)
    save_fn = model_desc.get("save_fn", _nop)
    test_fn = model_desc["test_fn"]

    train_results = []
    test_results = []
    for i in range(n):
        j = i + 1
        # get training_data
        train, validation, test = get_training_data(data_desc, seed + i, i=i, n=n, split=split)
        # create model
        model = create_model(i=i, seed=seed + i, atomizer=atomizer)
        # train model
        start = time.time()
        train_result = train_fn(model=model, train=train, validation=validation, i=i, seed=seed + i) or {}
        elapsed = time.time() - start
        train_result["time"] = elapsed
        m = len(train["y"])
        #if elapsed > 3:
        if not quiet:
            print("[{j:2} of {n}] training on {m} instances took {elapsed:.2f}s".format(**vars()))
        train_results.append(train_result)
        # save model
        save_fn(model=model, i=i, n=n, seed=seed + i)
        # make predictions
        start = time.time()
        predictions = test_fn(model=model, test=test, i=i, seed=seed + i)
        elapsed = time.time() - start
        test_result = analyze(predictions, test)
        test_result["time"] = elapsed
        accuracy = test_result["accuracy"]
        geo_speedup = test_result["speedup_geo"]
        m = len(test["y"])

        if not quiet:
            print("[{j:2} of {n}] accuracy on {m} test instances: {accuracy:.2%}, speedup {geo_speedup:.2}x (elapsed {elapsed:.2f}s)".format(**vars()))
        test_results.append(test_result)

        del model

    # accuracy
    a = np.array([x["accuracy"] for x in test_results])
    mean_acc = a.mean()
    std = a.std()

    a = np.array([x["speedup_geo"] for x in test_results])
    geo_speedup = gmean(a)
    std_speedup = a.std()

    # times
    a = np.array([x["time"] for x in train_results])
    mean_train_time = a.mean()
    std_train_time = a.std()

    a = np.array([x["time"] for x in test_results])
    mean_eval_time = a.mean()
    std_eval_time = a.std()

    if not quiet:
        print()
    print("avg training time {mean_train_time:.2f}s, eval time {mean_eval_time:.2f}s".format(**vars()))
    print("avg accuracy {mean_acc:.2%} (std: {std:.2%}), oracle {geo_speedup:.2f}x (std: {std_speedup:.2%})".format(**vars()))
    return train_results, test_results


def experiments(model_desc, evaluate_opts={}, dataframe_opts={}):
    r = []
    for a in [["amd", "B"], ["nvidia", "B"]]:
        platform, source = a
        print("PLATFORM", platform, "SOURCE", source)
        data_desc = load_dataframe(platform=platform, source=source, **dataframe_opts)
        r.append(evaluate(data_desc, model_desc, **evaluate_opts))
        print()


### UTILITY

def get_model_path(model_desc, platform, source, split, atomizer=CharacterAtomizer, maxlen=1024, seed=204):
    data_desc = load_dataframe(platform=platform, source=source,
                               max_seq_len=maxlen, atomizer=atomizer)

    split_txt = ":".join("{:02d}".format(round(d * 100)) for d in split)
    atomizer_txt = type(data_desc["atomizer"]).__name__
    name = model_desc["name"]
    return "models/{name}/{platform}-{source}-{split_txt}-{atomizer_txt}:{maxlen}-{seed}.model".format(**vars())

def train_and_save(model_desc, platform, source,
                   split=(.6, .2, .2), atomizer=CharacterAtomizer, maxlen=1024,
                   seed=204):
    np.random.seed(seed)

    name = model_desc["name"]
    create_fn = model_desc.get("create_model", _nop)
    train_fn = model_desc.get("train_fn", _nop)
    save_fn = model_desc["save_fn"]

    # create data split
    data_desc = load_dataframe(platform=platform, source=source,
                               max_seq_len=maxlen, atomizer=atomizer)
    train, validation, test = get_training_data(data_desc, seed, split=split)

    # create model
    model = create_fn(seed=seed, data_desc=data_desc)

    # train model
    train_fn(model=model, train=train, validation=validation, seed=seed)

    fs.mkdir("models/{name}".format(**vars()))
    split_txt = ":".join("{:02d}".format(round(d * 100)) for d in split)
    atomizer_txt = type(data_desc["atomizer"]).__name__
    outpath = "models/{name}/{platform}-{source}-{split_txt}-{atomizer_txt}:{maxlen}-{seed}.model".format(**vars())
    save_fn(outpath, model)
    print("model saved as", outpath)

def load_and_test(model_desc, platform, source,
                  split=(.6, .2, .2), atomizer=CharacterAtomizer, maxlen=1024,
                  seed=204):
    np.random.seed(seed)

    name = model_desc["name"]
    test_fn = model_desc["test_fn"]
    load_fn = model_desc["load_fn"]

    # load training data
    data_desc = load_dataframe(platform=platform, source=source,
                               max_seq_len=maxlen, atomizer=atomizer,
                               quiet=True)
    train, validation, test = get_training_data(data_desc, seed, split=split)

    # load model
    split_txt = ":".join("{:02d}".format(round(d * 100)) for d in split)
    atomizer_txt = type(data_desc["atomizer"]).__name__
    inpath = "models/{name}/{platform}-{source}-{split_txt}-{atomizer_txt}:{maxlen}-{seed}.model".format(**vars())
    model = load_fn(inpath)
    print("model loaded from", inpath)

    # test model
    predictions = test_fn(model=model, test=test, seed=seed)
    test_result = analyze(predictions, test)

    return test_result
