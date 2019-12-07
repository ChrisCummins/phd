"""A module for encoding node embeddings."""
import numpy as np
import pickle
from typing import List

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import decorators

FLAGS = app.FLAGS

AUGMENTED_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings/inst2vec_augmented_embeddings.pickle"
)


class GraphEncoder(object):
  @decorators.memoized_property
  def embeddings_tables(self) -> List[np.array]:
    """Return the embeddings tables."""
    # TODO(github.com/ChrisCummins/ProGraML/issues/12): In the future we may
    # want to add support for different numbers of embeddings tables, or
    # embeddings tables with different types. This is hardcoded to support only
    # two embeddings tables: our augmented inst2vec statement embeddings, and
    # a binary 'selector' table which can be used to select one or more nodes
    # of interests in graphs, e.g. for setting the starting point for computing
    # iterative data flow analyses.
    with open(AUGMENTED_INST2VEC_EMBEDDINGS, "rb") as f:
      augmented_inst2vec_embeddings = pickle.load(f)

    node_selector = np.vstack([[1, 0], [0, 1],]).astype(np.float64)

    return [augmented_inst2vec_embeddings, node_selector]
