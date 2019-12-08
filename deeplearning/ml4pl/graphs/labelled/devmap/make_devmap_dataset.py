"""This module prepares a CPU/GPU OpenCL device-mapping dataset."""
from typing import Iterable

import numpy as np
import pandas as pd
import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple as graph_tuples
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.ir import ir_database
from labm8.py import app
from labm8.py import progress
from labm8.py import sqlutil

app.DEFINE_string(
  "gpu", None, "The gpu to use. One of: {amd_tahiti_7970,nvidia_gtx_960}"
)

FLAGS = app.FLAGS


def MakeGpuDataFrame(df: pd.DataFrame, gpu: str):
  """Construct a labelled data frame for the given GPU, where the 'y' column
  indicates whether the GPU was faster than the CPU.
  """
  cpu = "intel_core_i7_3820"

  df["relpath"] = [
    ":".join(
      [
        r["program:benchmark_suite_name"],
        r["program:benchmark_name"],
        r["program:opencl_kernel_name"],
      ]
    )
    for _, r in df.iterrows()
  ]

  df["y"] = [
    np.array([0, 1], dtype=np.int32)
    if r[f"runtime:{gpu}"] < r[f"runtime:{cpu}"]
    else np.array([1, 0], dtype=np.int32)
    for _, r in df.iterrows()
  ]

  df.rename(
    columns={
      f"param:{gpu}:wgsize": "wgsize",
      f"feature:{gpu}:transfer": "transfer",
      f"runtime:{cpu}": "runtime_cpu",
      f"runtime:{gpu}": "runtime_gpu",
    },
    inplace=True,
  )

  return df[
    [
      "relpath",
      "wgsize",
      "transfer",
      "y",
      "runtime_cpu",
      "runtime_gpu",
      "data:dataset_name",
      "program:opencl_src",
    ]
  ]


def AnnotateGraphMetas(
  ir_db: ir_database.Database,
  proto_db: unlabelled_graph_database.Database,
  df: pd.DataFrame,
  ctx: progress.ProgressContext = progress.NullContext,
) -> Iterable[graph_tuple_database.GraphTuple]:
  """Add features and labels to graph metas in database."""
  with ir_db.Session() as ir_session, proto_db.Session() as proto_session:
    for _, row in df.iterrows():
      relpath = row["relpath"]
      with ctx.Profile(
        2, f"Processed graph {row['relpath']}:{row['data:dataset_name']}"
      ):
        # Select the corresponding IR.
        ir_id = (
          ir_session.query(ir_database.IntermediateRepresentation.id)
          .filter(
            ir_database.IntermediateRepresentation.source
            == "pact17_opencl_devmap",
            ir_database.IntermediateRepresentation.relpath == relpath,
          )
          .scalar()
        )
        # Check that we have an exact 1:1 mapping from the opencl devmap dataset
        # to IR.
        if ir_id is None:
          raise ValueError(f"Expected one IR with relpath {relpath}")

        # Load the program graph.
        proto_row = (
          proto_session.query(unlabelled_graph_database.ProgramGraph)
          .filter(unlabelled_graph_database.ProgramGraph.ir_id == ir_id)
          .options(
            sql.orm.joinedload(unlabelled_graph_database.ProgramGraph.data)
          )
          .scalar()
        )
        if proto_row is None:
          raise ValueError(
            f"Expected one proto for relpath {relpath} with ID {ir_id}"
          )
        proto: programl_pb2.ProgramGraph = proto_row.proto

        # Add the null "selector vector" value.
        for node in proto.node:
          node.x.append(0)

        # Add the graph-level features.
        proto.x[:] = [row["wgsize"], row["transfer"]]
        # Add 'y' graph feature as target.
        proto.y[:] = row["y"].tolist()

        # Create the graph tuple. Note the jumping through hoops with converting
        # proto -> nx -> graph_tuple, because there is currently no direct
        # proto -> graph_tuple conversion.
        graph_tuple = graph_tuple_database.GraphTuple.CreateFromGraphTuple(
          graph_tuple=graph_tuples.GraphTuple.CreateFromNetworkX(
            programl.ProgramGraphToNetworkX(proto)
          ),
          ir_id=ir_id,
        )
      yield graph_tuple


class MakeOpenClDevmapDataset(progress.Progress):
  """Create a labelled dataset for the given GPU."""

  def __init__(
    self,
    ir_db: ir_database.Database,
    proto_db: unlabelled_graph_database.Database,
    graph_db: graph_tuple_database.Database,
    gpu: str,
  ):
    self.ir_db = ir_db
    self.proto_db = proto_db
    self.graph_db = graph_db
    self.gpu = gpu

    self.dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
    super(MakeOpenClDevmapDataset, self).__init__(
      gpu, i=0, n=len(self.dataset.df), unit="protos"
    )

  def Run(self):
    with sqlutil.BufferedDatabaseWriter(
      self.graph_db, max_buffer_size=32 * 1024 * 1024, max_buffer_length=1024
    ) as writer:
      df = MakeGpuDataFrame(self.dataset.df, self.gpu)

      for graph in AnnotateGraphMetas(
        self.ir_db, self.proto_db, df, ctx=self.ctx
      ):
        self.ctx.i += 1
        writer.AddOne(graph)


def main():
  """Main entry point."""
  ir_db = FLAGS.ir_db()
  proto_db = FLAGS.proto_db()
  graph_db = FLAGS.graph_db()
  gpu = FLAGS.gpu

  if gpu not in {"amd_tahiti_7970", "nvidia_gtx_960"}:
    raise app.UsageError("Unknown GPU")

  progress.Run(MakeOpenClDevmapDataset(ir_db, proto_db, graph_db, gpu))


if __name__ == "__main__":
  app.Run(main)
