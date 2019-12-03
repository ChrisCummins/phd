"""A database of unlabelled ProGraML ProgramGraph protocol buffers."""
import datetime
import pickle
from typing import Any
from typing import Optional

import sqlalchemy as sql

from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = sql.ext.declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """A key-value database metadata store, with additional run ID."""

  # Unused integer ID for this row.
  id: int = sql.Column(sql.Integer, primary_key=True)

  # The run ID that generated this <key,value> pair.
  run_id: str = run_id.RunId.SqlStringColumn()

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # The <key,value> pair.
  key: str = sql.Column(sql.String(128), index=True)
  pickled_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @property
  def value(self) -> Any:
    """De-pickle the column value."""
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: Any):
    """Construct a table entry."""
    return Meta(key=key, pickled_value=pickle.dumps(value))


class ProgramGraph(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A table of ProGraML program graphs.

  For every ProgramGraph, there should be a corresponding ProgramGraphData row
  containing the encoded protocol buffer as a binary blob. The reason for
  dividing the data horizontally across two tables is to enable fast scanning
  of proto metadata, without needing to churn through a table of binary proto
  strings.
  """

  # A reference to the 'id' column of a
  # deeplearning.ml4pl.ir.ir_database.IntermediateRepresentationFile database
  # row. There is no foreign key relationship here because they are separate
  # databases. There is a one-to-one mapping from intermediate representation
  # to ProgramGraph.
  ir_id: int = sql.Column(sql.Integer, primary_key=True)

  # An integer used to split databases of graphs into separate graphs, e.g.
  # train/val/test split.
  split: Optional[int] = sql.Column(sql.Integer, nullable=True, index=True)

  # The size of the program graph.
  node_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)

  # The number of distinct node types and edge flows, respectively.
  node_type_count: int = sql.Column(sql.Integer, nullable=False)
  edge_flow_count: int = sql.Column(sql.Integer, nullable=False)

  # The number of unique {text, preprocessed_text} attributes.
  node_unique_text_count: int = sql.Column(sql.Integer, nullable=False)
  node_unique_preprocessed_text_count: int = sql.Column(
    sql.Integer, nullable=False
  )

  # The dimensionality of graph-level {x, y} vectors.
  graph_x_dimensionality: int = sql.Column(
    sql.Integer, nullable=False, default=0
  )
  graph_y_dimensionality: int = sql.Column(
    sql.Integer, nullable=False, default=0
  )

  # The dimensionality of node-level {x, y} vectors.
  node_x_dimensionality: int = sql.Column(
    sql.Integer, nullable=False, default=0
  )
  node_y_dimensionality: int = sql.Column(
    sql.Integer, nullable=False, default=0
  )

  # The maximum value of the 'position' attribute of edges.
  edge_position_max: int = sql.Column(sql.Integer, nullable=False)

  # The size of the serialized proto in bytes.
  serialized_proto_size: int = sql.Column(sql.Integer, nullable=False)

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # Create the one-to-one relationship from ProgramGraphs to ProgramGraphData.
  data: "ProgramGraphData" = sql.orm.relationship(
    "ProgramGraphData", uselist=False, cascade="all, delete-orphan"
  )

  # Joined table accessors:

  @property
  def sha1(self) -> str:
    """Return the sha1 of the serialized proto."""
    return self.data.sha1

  @property
  def proto(
    self, proto: programl_pb2.ProgramGraph = None
  ) -> programl_pb2.ProgramGraph:
    """Deserialize and load the protocol buffer."""
    proto = proto or programl_pb2.ProgramGraph()
    proto.ParseFromString(self.data.serialized_proto)
    return proto

  @classmethod
  def Create(
    cls,
    proto: programl_pb2.ProgramGraph,
    ir_id: int,
    split: Optional[int] = None,
  ) -> "ProgramGraph":
    """Create a ProgramGraph from the given protocol buffer.

    This is the preferred method of populating databases of program graphs, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two proto/metadata invisibly.

    Args:
      proto: The protocol buffer to instantiate a program graph from.
      ir_id: The ID of the intermediate representation for this program graph.
      split: The split of the proto buf.

    Returns:
      A ProgramGraph instance.
    """
    # Gather the edge attributes in a single pass of the proto.
    edge_attributes = [(edge.flow, edge.position) for edge in proto.edge]
    edge_flows = set([x[0] for x in edge_attributes])
    edge_position_max = max([x[1] for x in edge_attributes])
    del edge_attributes

    # Gather the node attributes in a single pass.
    node_types = set()
    node_texts = set()
    node_preprocessed_texts = set()
    node_x_dimensionalities = set()
    node_y_dimensionalities = set()

    for node in proto.node:
      node_types.add(node.type)
      node_texts.add(node.text)
      node_preprocessed_texts.add(node.preprocessed_text)
      node_x_dimensionalities.add(len(node.x))
      node_y_dimensionalities.add(len(node.y))

    if len(node_x_dimensionalities) != 1:
      raise ValueError(
        "Graph contains multiple node-level x dimensionalities: "
        f"{node_x_dimensionalities}"
      )
    if len(node_y_dimensionalities) != 1:
      raise ValueError(
        "Graph contains multiple node-level y dimensionalities: "
        f"{node_y_dimensionalities}"
      )

    serialized_proto = proto.SerializeToString()

    return ProgramGraph(
      ir_id=ir_id,
      split=split,
      node_count=len(proto.node),
      edge_count=len(proto.edge),
      node_type_count=len(node_types),
      edge_flow_count=len(edge_flows),
      node_unique_text_count=len(node_texts),
      node_unique_preprocessed_text_count=len(node_preprocessed_texts),
      graph_x_dimensionality=len(proto.x),
      graph_y_dimensionality=len(proto.y),
      node_x_dimensionality=list(node_x_dimensionalities)[0],
      node_y_dimensionality=list(node_y_dimensionalities)[0],
      edge_position_max=edge_position_max,
      serialized_proto_size=len(serialized_proto),
      data=ProgramGraphData(
        sha1=crypto.sha1(serialized_proto), serialized_proto=serialized_proto,
      ),
    )


class ProgramGraphData(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """The protocol buffer of a program graph.

  See ProgramGraph for the parent table.
  """

  ir_id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey(
      "program_graphs.ir_id", onupdate="CASCADE", ondelete="CASCADE"
    ),
    primary_key=True,
  )

  # The sha1sum of the 'serialized_proto' column. There is no requirement
  # that unlabelled graphs be unique, but, should you wish to enforce this,
  # you can group by this sha1 column and prune the duplicates.
  sha1: str = sql.Column(sql.String(40), nullable=False, index=True)

  # A binary-serialized ProgramGraph protocol buffer.
  serialized_proto: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )


class Database(sqlutil.Database):
  """A database of ProgramGraph protocol buffers."""

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
