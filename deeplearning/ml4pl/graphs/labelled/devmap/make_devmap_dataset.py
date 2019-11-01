"""This module prepares a CPU/GPU OpenCL device-mapping dataset."""
import pathlib
import tempfile
import typing

import numpy as np
import pandas as pd
import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_tuple as graph_tuples
from deeplearning.ncc.inst2vec import api as inst2vec
from labm8 import app
from labm8 import fs
from labm8 import sqlutil

app.DEFINE_database(
    'input_db',
    graph_database.Database,
    None,
    'URL of database to read unlabelled graphs from.',
    must_exist=True)
app.DEFINE_database('output_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write labelled graphs to.')
app.DEFINE_string('gpu', None,
                  'The gpu to use. One of: {amd_tahiti_7970,nvidia_gtx_960}')

FLAGS = app.FLAGS


def MakeGpuDataFrame(df: pd.DataFrame, gpu: str):
  """Construct a labelled data frame for the given GPU, where the 'y' column
  indicates whether the GPU was faster than the CPU.
  """
  cpu = 'intel_core_i7_3820'

  df['relpath'] = [
      ":".join([
          r['program:benchmark_suite_name'], r['program:benchmark_name'],
          r['program:opencl_kernel_name']
      ]) for _, r in df.iterrows()
  ]

  df['y'] = [
      np.array([0, 1], dtype=np.int32)
      if r[f'runtime:{gpu}'] < r[f'runtime:{cpu}'] else np.array([1, 0],
                                                                 dtype=np.int32)
      for _, r in df.iterrows()
  ]

  df.rename(
      columns={
          f'param:{gpu}:wgsize': 'wgsize',
          f'feature:{gpu}:transfer': 'transfer',
          f'runtime:{cpu}': 'runtime_cpu',
          f'runtime:{gpu}': 'runtime_gpu',
      },
      inplace=True)

  return df[[
      'relpath',
      'wgsize',
      'transfer',
      'y',
      'runtime_cpu',
      'runtime_gpu',
      'data:dataset_name',
  ]]


def MakeAnnotatedGraphs(input_db: graph_database.Database, df: pd.DataFrame
                       ) -> typing.Iterable[graph_database.GraphMeta]:
  """Make annotated graph's for the given devmap dataset."""
  with input_db.Session() as session:
    for _, row in df.iterrows():
      q = session.query(graph_database.GraphMeta)

      # Select the corresponding graph from the input database.
      q = q.filter(
          graph_database.GraphMeta.source_name == 'pact17_opencl_devmap')
      q = q.filter(graph_database.GraphMeta.relpath == row['relpath'])

      # Check that we have an exact 1:1 mapping from the opencl devmap dataset
      # to graphs in the input database.
      if q.count() != 1:
        app.Error("Expected one graph with relpath %s, but found %s",
                  row['relpath'], q.count())
        continue

      # Load the graph data.
      q = q.options(sql.orm.joinedload(graph_database.GraphMeta.graph))
      input_graph_meta = q.first()
      graph_tuple = input_graph_meta.data

      # Add the graph-level features.
      # TODO(cec): Should we apply any transforms to these values? Log?
      graph_x = np.array([row['wgsize'], row['transfer']], dtype=np.int64)

      # Add 'y' graph feature as target.
      graph_y = row['y']

      yield graph_database.GraphMeta.CreateFromGraphMetaAndGraphTuple(
          input_graph_meta,
          graph_tuples.GraphTuple(
              adjacency_lists=graph_tuple.adjacency_lists,
              edge_positions=graph_tuple.edge_positions,
              incoming_edge_counts=graph_tuple.incoming_edge_counts,
              node_x_indices=graph_tuple.node_x_indices,
              graph_x=graph_x,
              graph_y=graph_y,
          ))


def MakeOpenClDevmapDataset(input_db: graph_database.Database,
                            output_db: graph_database.Database, gpu: str):
  """Create a labelled dataset for the given GPU."""
  # TODO(cec): Change groups to k-fold classification.
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()

  with sqlutil.BufferedDatabaseWriter(
      output_db, max_queue=8).Session() as writer:
    df = MakeGpuDataFrame(dataset.df, gpu)

    for graph in MakeAnnotatedGraphs(input_db, df):
      app.Log(1, 'Processed graph')
      writer.AddOne(graph)


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()
  gpu = FLAGS.gpu

  if gpu not in {'amd_tahiti_7970', 'nvidia_gtx_960'}:
    raise app.UsageError("Unknown GPU")

  # Temporarily redirect logs to a file, which we will later import into the
  # database's meta table.
  with tempfile.TemporaryDirectory() as d:
    app.LogToDirectory(d, 'log')

    MakeOpenClDevmapDataset(input_db, output_db, gpu)

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with output_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
