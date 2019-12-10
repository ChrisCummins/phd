"""Generate labelled program graphs using data flow analysis.

This program reads a ProgramGraph protocol buffer from stdin, creates labelled
graphs by running a specified analysis, then prints the results as a
ProgramGraphs protocol buffer to stdout. Use --stdin_fmt and --stdout_fmt to
support binary or text formats.

List the available analyses using:

    $ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:annotate -- --list

For analyses that can produce multiple labelled graphs (e.g. by picking
different root nodes), use the --n argument to limit the number of generated
graphs.

For example, to produce up to 5 labelled graphs using reachability analysis
and text format protocol buffers:

    $ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:annotate -- \
        --analysis=reachability \
        --stdin_fmt=pbtxt \
        --stdout_fmt=pbtxt \
        --n=5 \
        < /tmp/program_graph.pbtxt
"""
import subprocess
import sys
import time
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled.dataflow.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.dataflow.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.dataflow.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.dataflow.polyhedra import polyhedra
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.graphs.labelled.dataflow.subexpressions import (
  subexpressions,
)
from deeplearning.ml4pl.testing import test_annotators
from labm8.py import app
from labm8.py import bazelutil


class TimeoutAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> Iterable[programl_pb2.ProgramGraph]:
    time.sleep(int(1e6))


# A map from analysis name to a callback which instantiates a
# DataFlowGraphAnnotator object for this anlysis. To add a new analysis, create
# a new entry in this table.
ANALYSES: Dict[str, Callable[[], data_flow_graphs.DataFlowGraphAnnotator]] = {
  "reachability": lambda: reachability.ReachabilityAnnotator(),
  # Annotators which are used for testing this script. These should, for obvious
  # reasons, not be used in prod. However, they must remain here so that we can
  # test the behaviour of the annotator under various conditions.
  "test_pass_thru": lambda: test_annotators.PassThruAnnotator(),
  "test_flaky": lambda: test_annotators.FlakyAnnotator(),
  "test_timeout": lambda: test_annotators.TimeoutAnnotator(),
  "test_error": lambda: test_annotators.ErrorAnnotator(),
}

# A list of the available analyses. We filter out the test_xxx named annotators
# for clarity.
AVAILABLE_ANALYSES = sorted(
  analysis for analysis in ANALYSES if not analysis.startswith("test_")
)

# The path of this script. Because a target cannot depend on itself, all calling
# code must add this script to its `data` dependencies.
SELF = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/labelled/dataflow/annotate"
)

app.DEFINE_boolean(
  "list", False, "If true, list the available analyses and exit."
)
app.DEFINE_string("analysis", "", "The name of the analysis to run.")
app.DEFINE_integer(
  "n",
  0,
  "The maximum number of labelled program graphs to produce. "
  "For a graph with `n` root statements, `n` instances can be produced by "
  "changing the root statement. If --n=0, enumerate all possible labelled "
  "graphs.",
)

FLAGS = app.FLAGS


class AnalysisFailed(ValueError):
  """An error raised if the analysis failed."""

  def __init__(self, returncode: int, stderr: str):
    self.returncode = returncode
    self.stderr = stderr

  def __repr__(self) -> str:
    return f"Analysis failed: {self.stderr}"

  def __str__(self) -> str:
    return repr(self)


class AnalysisTimeout(AnalysisFailed):
  def __init__(self, returncode: int, stderr: str, timeout: int):
    super(AnalysisTimeout, self).__init__(returncode, stderr)
    self.timeout = timeout

  def __repr__(self) -> str:
    return f"Analysis failed to complete within {self.timeout} seconds"

  def __str__(self) -> str:
    return repr(self)


# Return codes for error conditions.
#
# Error initializing the requested analysis.
E_ANALYSIS_INIT = 10
# Error reading stdin.
E_INVALID_INPUT = 11
# The analysis failed.
E_ANALYSIS_FAILED = 12
# Error writing stdout.
E_INVALID_STDOUT = 13


def Annotate(
  analysis: str,
  graph: Union[programl_pb2.ProgramGraph, bytes],
  n: int = 0,
  timeout: int = 120,
  binary_graph: bool = False,
) -> programl_pb2.ProgramGraphs:
  """Programatically run this script and return the output.

  DISCLAIMER: Because a target cannot depend on itself, all calling code must
  add //deeplearning/ml4pl/graphs/labelled/dataflow:annotate to its list of
  data dependencies.

  Args:
    analysis: The name of the analysis to run.
    graph: The unlabelled ProgramGraph protocol buffer to to annotate, either
      as a proto instance or as binary-encoded byte array.
    n: The maximum number of labelled graphs to produce.
    timeout: The maximum number of seconds to run the analysis for.
    binary_graph: If true, treat the graph argument as a binary byte array.

  Returns:
    A ProgramGraphs protocol buffer.

  Raises:
    IOError: If serializing the input or output protos fails.
    ValueError: If an invalid analysis is requested.
    AnalysisFailed: If the analysis raised an error.
    AnalysisTimeout: If the analysis did not complete within the requested
      timeout.
  """
  process = subprocess.Popen(
    [
      "timeout",
      "-s9",
      str(timeout),
      str(SELF),
      "--analysis",
      analysis,
      "--n",
      str(n),
      "--stdin_fmt",
      "pb",
      "--stdout_fmt",
      "pb",
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Encode the input if required.
  if binary_graph:
    stdin = graph
  else:
    stdin = programl.ToBytes(graph, fmt=programl.InputOutputFormat.PB)

  stdout, stderr = process.communicate(stdin)
  if process.returncode == 9 or process.returncode == -9:
    # Process was killed. We assume this is because of timeout, though it could
    # be the user.
    raise AnalysisTimeout(process.returncode, stderr, timeout)
  elif process.returncode == E_INVALID_INPUT:
    raise IOError("Failed to serialize input graph")
  elif process.returncode == E_INVALID_STDOUT:
    raise IOError("Analysis failed to write stdout")
  elif process.returncode == E_ANALYSIS_INIT:
    raise ValueError(stderr.decode("utf-8"))
  elif process.returncode:
    raise AnalysisFailed(process.returncode, stderr.decode("utf-8"))

  # Construct the protocol buffer from stdout.
  return programl.FromBytes(
    stdout,
    programl.InputOutputFormat.PB,
    proto=programl_pb2.ProgramGraphs(),
    empty_okay=True,
  )


def Main():
  """Main entry point."""
  if FLAGS.list:
    print(f"Available analyses: {AVAILABLE_ANALYSES}")
    return

  n = FLAGS.n

  try:
    annotator = ANALYSES.get(FLAGS.analysis, lambda: None)()
  except Exception as e:
    print(f"Error initializing analysis: {e}", file=sys.stderr)
    sys.exit(E_ANALYSIS_INIT)

  if not annotator:
    print(
      f"Unknown analysis: {FLAGS.analysis}. "
      f"Available analyses: {AVAILABLE_ANALYSES}",
      file=sys.stderr,
    )
    sys.exit(E_ANALYSIS_INIT)

  try:
    input_graph = programl.ReadStdin()
  except Exception as e:
    print(f"Error parsing stdin: {e}")
    sys.exit(E_INVALID_INPUT)

  annotated_graphs: List[programl_pb2.ProgramGraph] = []
  try:
    for annotated_graph in annotator.MakeAnnotated(input_graph, n):
      annotated_graphs.append(annotated_graph)
  except Exception as e:
    print(f"Error during analysis: {e}")
    sys.exit(E_ANALYSIS_FAILED)

  try:
    programl.WriteStdout(programl_pb2.ProgramGraphs(graph=annotated_graphs))
  except Exception as e:
    print(f"Error writing stdout: {e}")
    sys.exit(E_INVALID_STDOUT)


if __name__ == "__main__":
  app.Run(Main)