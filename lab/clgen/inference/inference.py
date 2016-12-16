# Usage:
#
# python ./inference.py ~/model-512x2x50.json 2>&1 | tee inference-model-512x2x50.log
#
import sys

from collections import Counter
from time import time
from labm8 import fs
from labm8 import system
from labm8.time import nowstr

import clgen
from clgen import clutil
from clgen import corpus
from clgen import dbutil
from clgen import log
from clgen import sampler
from clgen import model
from clgen import preprocess


def evaluate(model, sampler):
    """ evaluate sampling efficiency """

    print("starting sampling")
    sampler.sample(model)

    print("preprocessing sample")
    sample_db = sampler.cache(model)["kernels.db"]
    preprocess.preprocess_db(sample_db)

    num_kernels = dbutil.num_rows_in(sample_db, "ContentFiles")
    num_good_kernels = dbutil.num_good_kernels(sample_db)
    num_ugly_kernels = dbutil.num_rows_in(sample_db, "PreprocessedFiles",
                                          "WHERE status=2")
    discard_rate = 1 - (num_good_kernels / num_kernels)
    ugly_rate = 1 - (num_ugly_kernels / num_kernels)

    total_charcount = dbutil.cc(sample_db, "ContentFiles")
    good_charcount = dbutil.cc(sample_db, "PreprocessedFiles",
                               condition="WHERE status=0")

    return {
        "argspec": sampler.kernel_opts["args"],
        "num_kernels": num_kernels,
        "num_good_kernels": num_good_kernels,
        "discard_rate": discard_rate,
        "ugly_rate": ugly_rate,
        "total_charcount": total_charcount,
        "good_charcount": good_charcount,
        "corpus_dir": model.corpus.cache.path,
        "model_dir": model.cache.path,
        "sampler_dir": sampler.cache(model).path,
    }

def main():
    log.init(verbose=True)

    m = model.from_json(clgen.load_json_file(sys.argv[1]))
    c = corpus.Corpus.from_json({"path": "~/data/github"})
    print("CLgen:      ", clgen.version())
    print("Corpus size:", c.size)
    print("Vocab size: ", c.vocab_size)

    m.train()

    p, _ = corpus.most_common_prototypes(c, 20)
    for i, row in enumerate(p):
        outpath = "./inference-p" + str(i + 1) + "-" + fs.basename(sys.argv[1])
        if fs.exists(outpath):
            continue

        _, prototype = row
        argspec = [' '.join(x.split()[:-1]) for x in prototype.split(',')]
        print("argspec", ','.join([str(x) for x in argspec]))
        s = sampler.from_json({
            "kernels": {
                "args": argspec,
                "max_length": 10000
            },
            "sampler": {
                "batch_size": 5000,
                "max_batches": 1,
                "static_checker": False,
                "dynamic_checker": False
            }
        })

        info = evaluate(m, s)
        clgen.write_file(outpath, clgen.format_json(info))


if __name__ == "__main__":
    main()
