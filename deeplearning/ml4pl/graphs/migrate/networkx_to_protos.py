"""Migrate networkx graphs to ProGraML protocol buffers.

See <github.com/ChrisCummins/ProGraML/issues/1>.
"""
import multiprocessing
import threading
from typing import Iterable
from typing import List

import networkx as nx
import sqlalchemy as sql
import tqdm

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled import programl_pb2
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import ppar
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_database(
  "input_db",
  graph_database.Database,
  None,
  "The database of unlabelled networkx graphs to migrate.",
  must_exist=True,
)
app.DEFINE_database(
  "output_db",
  unlabelled_graph_database.Database,
  None,
  "The program graph proto database to write results to.",
)
app.DEFINE_integer(
  "nproc", multiprocessing.cpu_count(), "The number of processes to spawn."
)


def NetworkXGraphToProgramGraphProto(
  g: nx.MultiDiGraph,
) -> programl_pb2.ProgramGraph:
  """Convert a networkx graph constructed using the old control-and-data-flow
  graph builder to a ProGraML graph proto."""
  proto = programl_pb2.ProgramGraph()

  # Create the map from nodes to function names.
  functions = set([fn for _, fn in g.nodes(data="function") if fn])
  function_to_idx_map = {fn: i for i, fn in enumerate(sorted(functions))}

  # Build a translation map from node names to node list indices.
  if "root" not in g.nodes:
    raise ValueError(f"Graph has no root node: {g.nodes}")
  node_to_idx_map = {"root": 0}
  for node in [node for node in g.nodes if node != "root"]:
    node_to_idx_map[node] = len(node_to_idx_map)

  # Create the node list.
  idx_to_node_map = {v: k for k, v in node_to_idx_map.items()}
  for node_idx in range(len(node_to_idx_map)):
    node = g.nodes[idx_to_node_map[node_idx]]
    node_proto = proto.node.add()

    # Translate node attributes.
    node_type = node.get("type")
    if not node_type:
      raise ValueError(f"Node has no type: {node_type}")
    node_proto.type = {
      "statement": programl_pb2.Node.STATEMENT,
      "identifier": programl_pb2.Node.IDENTIFIER,
      "immediate": programl_pb2.Node.IMMEDIATE,
      # We are removing the "magic" node type, replacing them with a regular
      # statement of unknown type.
      "magic": programl_pb2.Node.STATEMENT,
    }[node_type]

    # Get the text of the node.
    if "original_text" in node:
      node_proto.text = node["original_text"]
      node_proto.preprocessed_text = node["text"]
    elif "text" in node:
      node_proto.text = node["text"]
      node_proto.preprocessed_text = node["text"]
    elif "name" in node:
      node_proto.text = node["name"]
      node_proto.preprocessed_text = node["name"]
    else:
      raise ValueError(f"Node has no original_text or name: {node}")

    # Set the encoded representation of the node.
    x = node.get("x", None)
    if x is not None:
      node_proto.encoded = x

    # Set the node function.
    function = function_to_idx_map.get(node.get("function"))
    if function:
      node_proto.function = function

  # Create the function list.
  idx_to_function_map = {v: k for k, v in function_to_idx_map.items()}
  for fn_idx in range(len(function_to_idx_map)):
    function = proto.function.add()
    function.name = idx_to_function_map[fn_idx]

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge = proto.edge.add()
    edge.type = {
      "call": programl_pb2.Edge.CALL,
      "control": programl_pb2.Edge.CONTROL,
      "data": programl_pb2.Edge.DATA,
    }[data["flow"]]
    edge.source_node = node_to_idx_map[src]
    edge.destination_node = node_to_idx_map[dst]
    edge.position = data.get("position", 0)

  return proto


def BatchedGraphReader(
  graph_db: graph_database.Database,
  ids_to_read: List[int],
  batch_size: int,
  profiler,
) -> Iterable[List[graph_database.GraphMeta]]:
  """Read from the given list of graph IDs in batches."""
  ids_to_read = list(sorted(ids_to_read))
  with graph_db.Session() as session:
    for i in range(0, len(ids_to_read), batch_size):
      ids_batch = ids_to_read[i : i + batch_size]
      with profiler(f"[reader] Read {len(ids_batch)} input graphs"):
        graphs = session.query(graph_database.GraphMeta).filter(
          graph_database.GraphMeta.id >= ids_batch[0]
        ).filter(graph_database.GraphMeta.id <= ids_batch[-1]).options(
          sql.orm.joinedload(graph_database.GraphMeta.graph)
        ).all()
      yield graphs


def GraphMetaToProgramGraph(
  graph_meta: graph_database.GraphMeta,
) -> List[unlabelled_graph_database.ProgramGraph]:
  """Convert a list of graph metas to a list of program graphs."""
  graph: nx.MultiDiGraph = graph_meta.data
  proto = NetworkXGraphToProgramGraphProto(graph)
  program_graph = unlabelled_graph_database.ProgramGraph.Create(
    proto, split=graph_meta.group, ir_id=graph_meta.bytecode_id
  )
  program_graph.id = graph_meta.id
  return program_graph


class Migrator(threading.Thread):
  """Thread to migrate graph databases"""

  def __init__(
    self,
    graph_reader: ppar.ThreadedIterator,
    output_db: unlabelled_graph_database.Database,
    profiler,
    nproc: int,
    chunk_size: int,
    exported_count: int = 0,
  ):
    self.graph_reader = graph_reader
    self.output_db = output_db
    self.profiler = profiler
    self.nproc = nproc
    self.chunk_size = chunk_size
    self.exported_count = exported_count

    super(Migrator, self).__init__()
    self.start()

  def run(self):
    """Run the migration"""
    pool = multiprocessing.Pool(self.nproc)

    self.graph_reader.Start()

    # Main loop: Peel off chunks of graphs to process, process them, and write
    # the results back to the output database.
    for graph_batch in self.graph_reader:
      with self.output_db.Session() as out_session:
        # Divide the graphs into jobs for individual processes.
        workers = pool.imap_unordered(
          GraphMetaToProgramGraph, graph_batch, chunksize=self.chunk_size
        )
        # Record the generated program graphs.
        with self.profiler(f"[writer] Wrote {len(graph_batch)} program graphs"):
          for program_graph in workers:
            self.exported_count += 1
            out_session.add(program_graph)
          out_session.commit()


def MigrateGraphDatabase(
  input_db: graph_database.Database,
  output_db: unlabelled_graph_database.Database,
  nproc: int,
):
  """Migrate the entire contents of a networkx graph database to a proto
  database.
  """
  # Tunable parameters for dividing the workload. Graphs are read
  # asynchronously to a queue of maximum 'max_reader_queue_size' batches. Each
  # batch is of size 'batch_size', and each batch is divided across the
  # multiprocessing pool into chunks of size 'chunk_size'.
  max_reader_queue_size = 5
  batch_size = 256
  chunk_size = max(batch_size // 16, 1)

  # Setup: Get the IDs of the graphs to process.

  # Get the list of graphs that have already been processed.
  with output_db.Session() as out_session:
    already_done = set(
      [
        row.id
        for row in out_session.query(unlabelled_graph_database.ProgramGraph.id)
      ]
    )
    already_done_count = len(already_done)

  # Get the total number of graphs to process, and the number of graphs already
  # processed.
  with input_db.Session() as in_session:
    total_graph_count = in_session.query(
      sql.func.count(graph_database.GraphMeta.id)
    ).scalar()
    ids_to_do = [
      row.id
      for row in in_session.query(graph_database.GraphMeta.id)
      .filter(~graph_database.GraphMeta.id.in_(already_done))
      .order_by(graph_database.GraphMeta.id)
    ]

  # Sanity check.
  if len(ids_to_do) + already_done_count != total_graph_count:
    app.FatalWithoutStackTrace(
      "ids_to_do(%s) + already_done(%s) != total_rows(%s)",
      len(ids_to_do),
      already_done_count,
      total_graph_count,
    )

  app.Log(
    1, "Selected %s graphs to process", humanize.Plural(len(ids_to_do), "graph")
  )

  bar = tqdm.tqdm(
    initial=already_done_count,
    total=total_graph_count,
    desc="migrate",
    unit=" graphs",
    position=0,
  )
  # Wrap the profiling method so that it plays nicely with the progress bar.
  profiler = lambda msg: prof.Profile(
    msg,
    print_to=lambda msg: app.Log(1, msg, print_context=bar.external_write_mode),
  )

  graph_reader: Iterable[
    List[graph_database.GraphMeta]
  ] = ppar.ThreadedIterator(
    BatchedGraphReader(input_db, ids_to_do, batch_size, profiler=profiler),
    max_queue_size=max_reader_queue_size,
    start=False,
  )

  # Run migration asynchronously so that we can keep the progress bar updating.
  migrator = Migrator(
    graph_reader=graph_reader,
    output_db=output_db,
    profiler=profiler,
    nproc=nproc,
    chunk_size=chunk_size,
    exported_count=already_done_count,
  )
  while migrator.is_alive():
    bar.n = migrator.exported_count
    bar.refresh()
    migrator.join(0.25)

  app.Log(1, "Done!")


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()

  MigrateGraphDatabase(input_db, output_db, FLAGS.nproc)


if __name__ == "__main__":
  app.Run(main)
