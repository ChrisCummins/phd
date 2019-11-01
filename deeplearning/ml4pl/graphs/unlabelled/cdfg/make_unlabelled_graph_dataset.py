"""This module produces datasets of unlabelled graphs."""
import pathlib
import sys
import traceback
import typing

from labm8 import app

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import \
  make_data_flow_analysis_dataset as make_dataset
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS

BytecodeJob = typing.Tuple[str, str, str, str, int]


def _MakeBytecodeExportJob(session: bytecode_database.Database.SessionType,
                           bytecode_id: id) -> typing.Optional[BytecodeJob]:
  q = session.query(bytecode_database.LlvmBytecode.bytecode,
                    bytecode_database.LlvmBytecode.source_name,
                    bytecode_database.LlvmBytecode.relpath,
                    bytecode_database.LlvmBytecode.language) \
    .filter(bytecode_database.LlvmBytecode.id == bytecode_id).one()
  bytecode, source, relpath, language = q
  return bytecode, source, relpath, language, bytecode_id


def _ProcessBytecodeJob(
    job: BytecodeJob) -> typing.List[graph_database.GraphMeta]:
  """

  Args:
    job: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.

  Returns:
    A list containing a single graph.
  """
  bytecode, source_name, relpath, language, bytecode_id = job
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow='nodes_and_edges',
      preprocess_text=True,
      discard_unknown_statements=False,
  )

  try:
    graph = builder.Build(bytecode)
    graph.source_name = source_name
    graph.relpath = relpath
    graph.bytecode_id = str(bytecode_id)
    graph.language = language

    # Get the number of types of nodes and edges. We could do this more
    # efficiently by tracking these values during graph construction, which
    # would require a refactor.
    node_types = set([node_type for _, node_type in graph.nodes(data='type')])
    edge_type_count = set([flow for _, _, flow in graph.edges(data='flow')])

    if not len(node_types):
      raise ValueError("Graph has no nodes")
    if not len(edge_type_count):
      raise ValueError("Graph has no edges")

    graph_meta = graph_database.GraphMeta.CreateFromNetworkX(graph)
    app.Log(1, "Produced graph for bytecode %s with %s nodes",
            graph_meta.bytecode_id, graph_meta.node_count)
    return [graph_meta]
  except Exception as e:
    _, _, tb = sys.exc_info()
    tb = traceback.extract_tb(tb, 2)
    filename, line_number, function_name, *_ = tb[-1]
    filename = pathlib.Path(filename).name
    app.Error(
        'Failed to create control flow graphs from bytecode '
        '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
        function_name,
        type(e).__name__)
    return []


class BytecodeExporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export from LLVM bytecodes."""

  def __init__(self, *args, **kwargs):
    super(BytecodeExporter, self).__init__(*args, **kwargs)
    builder = cdfg.ControlAndDataFlowGraphBuilder(
        dataflow='nodes_and_edges',
        preprocess_text=True,
        discard_unknown_statements=False,
    )
    # TODO(cec): Fix.
    # self.graph_db.SetNodeEmbeddingTable(builder.embeddings_table)

  def GetMakeExportJob(self):
    return _MakeBytecodeExportJob

  def GetProcessJobFunction(
      self
  ) -> typing.Callable[[BytecodeJob], typing.List[graph_database.GraphMeta]]:
    return _ProcessBytecodeJob


def main():
  """Main entry point."""
  make_dataset.Run(BytecodeExporter)


if __name__ == '__main__':
  app.Run(main)
