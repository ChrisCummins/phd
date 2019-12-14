"""This module exposes a function for generating batch iterators."""
from typing import Dict
from typing import List

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import ppar
from labm8.py import progress


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


def MakeBatchIterator(
  model: classifier_base.ClassifierBase,
  splits: Dict[epoch.Type, List[int]],
  epoch_type: epoch.Type,
  ctx: progress.ProgressContext = progress.NullContext,
) -> batches.BatchIterator:
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
  if epoch_type == epoch.Type.TRAIN:
    limit = FLAGS.max_train_per_epoch
  elif epoch_type == epoch.Type.VAL:
    limit = FLAGS.max_val_per_epoch
  elif epoch_type == epoch.Type.TEST:
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

  graph_reader = graph_database_reader.BufferedGraphReader.CreateFromFlags(
    filters=[split_filter], ctx=ctx, limit=limit
  )

  return batches.BatchIterator(
    batches=ppar.ThreadedIterator(
      model.BatchIterator(graph_reader, ctx=ctx),
      max_queue_size=FLAGS.batch_queue_size,
    ),
    graph_count=graph_reader.n,
  )
