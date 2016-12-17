#!/usr/bin/env python3
#
# Usage:
#   ./search.py ~/src/clgen/model.json ~/src/clgen/tests/data/cl/sample-1.gs \
#       ~/src/clgen/tests/data/tiny/corpus/3.cl || less search.log
#
import labm8
import numpy as np

from labm8 import fs
from labm8.time import nowstr
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from random import randint

import clgen
from clgen import log
from clgen import model
from clgen import preprocess

if labm8.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


def features_from_file(path):
    """
    Fetch features from file.

    Arguments:
        path (str): Path to file.

    Returns:
        np.array: Feature values.
    """
    # hacky call to clgen-features and parse output
    cmd = ['clgen-features', path]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    cout, _ = proc.communicate()
    features = [float(x) for x in
                cout.decode('utf-8').split('\n')[1].split(',')[2:]]
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
        m.sample(seed_text=seed_text, output=buf, seed=seed, max_length=5000,
                 quiet=True)
        out = buf.getvalue()
        result = preprocess.preprocess(out)
        return 0, result
    except preprocess.BadCodeException:
        return 1, None
    except preprocess.UglyCodeException:
        return 2, None


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
    attempts = []

    max_attempts = 500

    for i in range(1, max_attempts + 1):
        # pick a random split and seed
        mutate_idx = randint(min_mutate_idx, max_mutate_idx)
        mutate_seed = randint(0, 255)

        start_text = start_code[:mutate_idx]

        print(">>> attempt", i, "idx", mutate_idx, "seed", mutate_seed, "-",
              end=" ")
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
            attempts.append((ret, mutate_idx, mutate_seed))
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


def search(m, start_code, target_code, logpath):
    log = []
    code_history = [start_code]

    if fs.dirname(logpath):
        print("mkdir", fs.dirname(logpath))
        fs.mkdir(fs.dirname(logpath))

    write_log(log, logpath)
    target_features = get_features(target_code)

    code = start_code
    features = get_features(code)
    distance = get_distance(target_features, features)

    add_to_log(log, {
        "start_code": start_code,
        "start_features": escape_features(features),
        "target_features": escape_features(target_features),
        "target_code": target_code,
        "distance": distance
    }, name="init")
    write_log(log, logpath)

    best = {
        "distance": distance,
        "idx": -1,
        "seed": 0,
        "code": code
    }

    # maximum number of mutations before stopping search
    max_count = 1000

    improved_counter = 0
    for i in range(max_count):
        newcode, mutate_idx, mutate_seed, attempts = get_mutation(m, code)

        entry = {
            "count": i,
            "attempts": attempts,
            "best": distance
        }

        if newcode:
            features = get_features(newcode)
            distance = get_distance(target_features, features)
            entry["code"] = newcode,
            entry["features"] = escape_features(features),
            entry["distance"] = distance
            entry["mutate_idx"] = mutate_idx,
            entry["mutate_seed"] = mutate_seed,
            code_history.append(code)
        else:
            print("step back")
            # step back
            if len(code_history):
                code = code_history.pop()
            entry["step_back"] = code

        if distance < best["distance"]:
            improved_counter += 1
            entry["improvement_count"] = improved_counter
            entry["improvement"] = -(1 - distance / best["distance"])

            best["distance"] = distance
            best["idx"] = mutate_idx
            best["seed"] = mutate_seed
            best["code"] = newcode

        add_to_log(log, entry, name="step")
        write_log(log, logpath)

        # doesn't have to be exactly zero but whatever
        if distance <= 0.001:
            print("found exact match!")
            break

    add_to_log(log, {
        "best_code": best['code'],
        "best_features": escape_features(best['features']),
        "best_distance": best['distance']
    }, name="end")
    write_log(log, logpath)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", help="Path to model")
    parser.add_argument("input", help="Path to starting code")
    parser.add_argument("target", help="Path to target code")
    parser.add_argument("-l", "--log", metavar="path", default="search-log.json",
                        help="Path to log file")
    args = parser.parse_args()

    log.init(verbose=True)

    # load model
    modelpath = args.model
    if modelpath.endswith(".tar.bz2"):
        m = model.from_tar(modelpath)
    else:
        model_json = clgen.load_json_file(modelpath)
        m = clgen.model.from_json(model_json)
    m.train()

    with open(args.input) as infile:
        start_code = infile.read()

    with open(args.target) as infile:
        target_code = infile.read()

    search(m, start_code, target_code, args.log)


if __name__ == "__main__":
    main()
