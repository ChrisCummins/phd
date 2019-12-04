"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import epoch
from labm8.py import app


FLAGS = app.FLAGS


class Checkpoint(NamedTuple):
  run_id: run_id_lib.RunId
  epoch_num: int
  best_results: Dict[epoch.Type, epoch.BestResults]
  model_data: Any


def RunIdAndEpochNumFromString(
  string: str,
) -> Tuple[run_id_lib.RunId, Optional[int]]:
  try:
    if "@" in string:
      # TODO: Split run_id@epoch_num:
      components = string.split("@")
      assert len(components) == 2
      run_id_string, epoch_num_string = components
      run_id = run_id_lib.RunId.FromString(run_id_string)
      epoch_num = int(epoch_num_string)
    else:
      run_id = run_id_lib.RunId.FromString(string)
      epoch_num = None

    return run_id, epoch_num
  except Exception:
    raise app.UsageError(f"Invalid run ID and epoch format: {string}")
