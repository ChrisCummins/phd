"""Unit tests for //deeplearning/ml4pl/graphs/labelled/graph_batcher."""
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs.labelled import make_data_flow_analysis_dataset

FLAGS = app.FLAGS


def test_GetAnnotatedGraphGenerators_unknown_analysis():
  with pytest.raises(app.UsageError):
    make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators('foo')


def test_GetAnnotatedGraphGenerators_with_requested_analyses():
  """Test requesting analyses by name."""
  annotators = make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
      'reachability', 'domtree')
  annotator_names = {a.name for a in annotators}

  assert annotator_names == {'reachability', 'domtree'}


if __name__ == '__main__':
  test.Main()
