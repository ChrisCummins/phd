# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module exposes a function for generating batch iterators."""
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from labm8.py import ppar
from labm8.py import progress
from programl.ml.batch.batch_data import BatchData
from programl.ml.epoch import epoch
from programl.ml.model import classifier_base

FLAGS = app.FLAGS

app.DEFINE_integer(
  "max_train_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "training epoch. For k-fold cross-validation, each of the k folds will "
  "train on a maximum of this many graphs.",
)
app.DEFINE_integer(
  "max_val_per_epoch",
  None,
  "Use this flag to limit the maximum number of instances used in a single "
  "validation epoch.",
)
app.DEFINE_integer(
  "batch_queue_size",
  10,
  "Tuning parameter. The maximum number of batches to generate before waiting "
  "for the model to complete. Must be >= 1.",
)


class BatchIterator(NamedTuple):
  """A batch iterator"""

  batches: Iterable[BatchData]
  # The total number of graphs in all of the batches.
  graph_count: int


def MakeBatchIterator(
  model: classifier_base.ClassifierBase,
  graph_db: graph_tuple_database.Database,
  splits: Dict[epoch.EpochType, List[int]],
  epoch_type: epoch.EpochType,
  ctx: progress.ProgressContext = progress.NullContext,
) -> BatchIterator:
  """Create an iterator over batches for the given epoch type.

  Args:
    model: The model to generate a batch iterator for.
    splits: A mapping from epoch type to a list of split numbers.
    epoch_type: The type of epoch to produce an iterator for.
    ctx: A logging context.

  Returns:
    A batch iterator for feeding into model.RunBatch().
  """
  # Filter the graph database to load graphs from the requested splits.
  if epoch_type == epoch.EpochType.TRAIN:
    limit = FLAGS.max_train_per_epoch
  elif epoch_type == epoch.EpochType.VAL:
    limit = FLAGS.max_val_per_epoch
  elif epoch_type == epoch.EpochType.TEST:
    limit = None  # Never limit the test set.
  else:
    raise NotImplementedError("unreachable")

  splits_for_type = splits[epoch_type]
  ctx.Log(
    3,
    "Using %s graph splits %s",
    epoch_type.name.lower(),
    sorted(splits_for_type),
  )

  if len(splits_for_type) == 1:
    split_filter = (
      lambda: graph_tuple_database.GraphTuple.split == splits_for_type[0]
    )
  else:
    split_filter = lambda: graph_tuple_database.GraphTuple.split.in_(
      splits_for_type
    )

  graph_reader = model.GraphReader(
    epoch_type=epoch_type,
    graph_db=graph_db,
    filters=[split_filter],
    limit=limit,
    ctx=ctx,
  )

  return BatchIterator(
    batches=ppar.ThreadedIterator(
      model.BatchIterator(epoch_type, graph_reader, ctx=ctx),
      max_queue_size=FLAGS.batch_queue_size,
    ),
    graph_count=graph_reader.n,
  )
