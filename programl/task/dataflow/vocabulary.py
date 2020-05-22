import csv
import pathlib

from labm8.py import app
from labm8.py import humanize


def LoadVocabulary(
  dataset_root: pathlib.Path,
  use_cdfg: bool,
  max_items: int = 0,
  target_cumfreq: float = 1.0,
):
  if use_cdfg:
    vocab_csv = dataset_root / "vocab" / "cdfg.csv"
  else:
    vocab_csv = dataset_root / "vocab" / "programl.csv"

  vocab = {}
  cumfreq = 0
  with open(vocab_csv) as f:
    vocab_file = csv.reader(f.readlines(), delimiter="\t")

    for i, row in enumerate(vocab_file, start=-1):
      if i == -1:  # Skip the header.
        continue
      (cumfreq, _, _, text) = row
      cumfreq = float(cumfreq)
      vocab[text] = i
      if cumfreq >= target_cumfreq:
        app.Log(2, "Reached target cumulative frequency: %.3f", target_cumfreq)
        break
      if max_items and i >= max_items - 1:
        app.Log(2, "Reached max vocab size: %d", max_items)
        break

  app.Log(
    1,
    "Selected %s-element vocabulary achieving %.2f%% node text coverage",
    humanize.Commas(len(vocab)),
    cumfreq * 100,
  )
  return vocab
