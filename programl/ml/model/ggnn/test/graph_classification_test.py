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
"""Unit tests for //deeplearning/ml4pl/models/ggnn."""
import math
from typing import List

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import test
from programl.ml.deprecated import batch_iterator as batch_iterator_lib
from programl.ml.epoch import epoch
from programl.ml.model import logger as logging
from programl.ml.model.ggnn import ggnn


FLAGS = test.FLAGS

# For testing models, always use --strict_graph_segmentation.
FLAGS.strict_graph_segmentation = True

FLAGS.label_conv_max_timesteps = 20

pytest_plugins = ["deeplearning.ml4pl.models.ggnn.test.fixtures"]


def CheckResultsProperties(
  results: epoch.EpochResults,
  graph_db: graph_tuple_database.Database,
  epoch_type: epoch.EpochType,
):
  """Check for various properties of well-formed epoch results."""
  assert isinstance(results, epoch.EpochResults)
  # Check that the epoch contains batches.
  assert results.batch_count

  # Check that results contain a loss value.
  assert results.has_loss
  assert results.loss
  assert not math.isnan(results.loss)

  # Test that the epoch included every graph (of the appropriate type) in the
  # database.
  assert results.graph_count == graph_db.split_counts[epoch_type.value]


@test.Parametrize("epoch_type", list(epoch.EpochType))
@test.Parametrize("limit_max_data_flow_steps", (False, True))
def test_graph_classifier_call(
  epoch_type: epoch.EpochType,
  logger: logging.Logger,
  layer_timesteps: List[str],
  graph_y_graph_db: graph_tuple_database.Database,
  limit_max_data_flow_steps: bool,
  node_text_embedding_type: str,
  unroll_strategy: str,
  log1p_graph_x: bool,
):
  """Run test epoch on a graph classifier."""
  FLAGS.inst2vec_embeddings = node_text_embedding_type
  FLAGS.unroll_strategy = unroll_strategy
  FLAGS.layer_timesteps = layer_timesteps
  FLAGS.log1p_graph_x = log1p_graph_x
  FLAGS.limit_max_data_flow_steps = limit_max_data_flow_steps

  # Test to handle the unsupported combination of config values.
  if (
    unroll_strategy == "label_convergence"
    and graph_y_graph_db.graph_x_dimensionality
  ) or (unroll_strategy == "label_convergence" and len(layer_timesteps) > 1):
    with test.Raises(AssertionError):
      ggnn.Ggnn(logger, graph_y_graph_db)
    return

  # Create and initialize an untrained model.
  model = ggnn.Ggnn(logger, graph_y_graph_db)
  model.Initialize()

  # Run the model over some random graphs.
  batch_iterator = batch_iterator_lib.MakeBatchIterator(
    model=model,
    graph_db=graph_y_graph_db,
    splits={
      epoch.EpochType.TRAIN: [0],
      epoch.EpochType.VAL: [1],
      epoch.EpochType.TEST: [2],
    },
    epoch_type=epoch_type,
  )

  results = model(
    epoch_type=epoch_type, batch_iterator=batch_iterator, logger=logger,
  )

  CheckResultsProperties(results, graph_y_graph_db, epoch_type)


if __name__ == "__main__":
  test.Main()
