# Usage:
#
# python ./benchmark.py model-64x2x50.json 2>&1 | tee benchmark-model-64x2x50.log
#
from time import time

import clgen
from clgen import dbutil
from clgen import log
from clgen import model
from clgen import preprocess
from clgen import sampler

from labm8.py import fs


def evaluate(model, sampler):
  """ evaluate sampling efficiency """
  model.cache.empty()  # clear checkpoint cache
  print("starting training")
  tstart = time()  # start timer
  model.train()  # train model
  training_time = time() - tstart

  # clear the sample cache
  sampler.cache(model).empty()

  # sample kernels and time
  print("starting sampling")
  tstart = time()
  sampler.sample(model)
  tend = time()
  elapsed = tend - tstart

  # preprocess sample
  sample_db = sampler.cache(model)["kernels.db"]
  preprocess.preprocess_db(sample_db)

  num_kernels = dbutil.num_rows_in(sample_db, "ContentFiles")
  num_good_kernels = dbutil.num_good_kernels(sample_db)
  num_ugly_kernels = dbutil.num_rows_in(sample_db, "PreprocessedFiles",
                                        "WHERE status=2")
  discard_rate = 1 - (num_good_kernels / num_kernels)
  ugly_rate = 1 - (num_ugly_kernels / num_kernels)

  total_charcount = dbutil.cc(sample_db, "ContentFiles")
  good_charcount = dbutil.cc(sample_db,
                             "PreprocessedFiles",
                             condition="WHERE status=0")

  efficiency = good_charcount / total_charcount
  throughput = good_charcount / elapsed

  return {
      "training_time": training_time,
      "sampling_time": elapsed,
      "num_kernels": num_kernels,
      "num_good_kernels": num_good_kernels,
      "discard_rate": discard_rate,
      "ugly_rate": ugly_rate,
      "total_charcount": total_charcount,
      "good_charcount": good_charcount,
      "efficiency": efficiency,  # good_chars / total_chars
      "throughput": throughput,  # good_chars / second
      "corpus_dir": model.corpus.cache.path,
      "model_dir": model.cache.path,
      "sampler_dir": sampler.cache(model).path,
  }


def main():
  import sys

  log.init(verbose=True)
  m = model.from_json(clgen.load_json_file(sys.argv[1]))
  s = sampler.from_json({
      "kernels": {
          "args": [
              "__global float*", "__global float*", "__global float*",
              "const int"
          ],
          "max_length":
          5000,
          "temperature":
          1
      },
      "sampler": {
          "batch_size": 1000,
          "max_batches": 1,
          "static_checker": False,
          "dynamic_checker": False
      }
  })

  print("Corpus size:", m.corpus.size)
  print("Vocab size: ", m.corpus.vocab_size)
  print()
  clgen.platform_info()
  print()

  outpath = "./benchmark-" + fs.basename(sys.argv[1])
  info = evaluate(m, s)
  clgen.write_file(outpath, clgen.format_json(info))


if __name__ == "__main__":
  main()
