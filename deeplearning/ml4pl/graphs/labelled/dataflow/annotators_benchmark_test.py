"""Benchmarks for comparing annotator performance."""
from deeplearning.ml4pl.graphs.labelled.dataflow import annotate
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import prof
from labm8.py import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = None

PYTEST_ARGS = ["--benchmark-warmup-iterations=2"]

# Real programs to test over.
PROTOS = list(random_programl_generator.EnumerateProtoTestSet(n=20))

# The annotators to test.
ANNOTATORS = {
  analysis: annotate.ANALYSES[analysis]
  for analysis in annotate.AVAILABLE_ANALYSES
}


def AnnotatorBenchmark(annotator_class):
  """A micro-benchmark that runs annotator over a list of test graphs."""
  with prof.Profile(
    f"Completed benchmark of {len(PROTOS) * 5} annotations "
    f"using {annotator_class.__name__}"
  ):
    for graph in PROTOS:
      annotator = annotator_class(graph)
      annotator.MakeAnnotated(5)


@test.Parametrize(
  "annotator", list(ANNOTATORS.values()), names=list(ANNOTATORS.keys())
)
def test_benchmark_nx_annotator(
  benchmark, annotator: data_flow_graphs.NetworkXDataFlowGraphAnnotator,
):
  """Benchmark analysis."""
  benchmark(AnnotatorBenchmark, annotator)


if __name__ == "__main__":
  test.Main()
