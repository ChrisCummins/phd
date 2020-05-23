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
"""Train an LSTM to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses an LSTM to predict binary
classification targets for data flow problems.
"""
import pathlib
import time
from typing import Dict

import numpy as np

from deeplearning.ncc import vocabulary
from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import ppar
from programl.ml.batch.async_batch_builder import AsyncBatchBuilder
from programl.ml.model.lstm.lstm import Lstm
from programl.proto import epoch_pb2
from programl.task.dataflow import dataflow
from programl.task.dataflow.graph_loader import DataflowGraphLoader
from programl.task.dataflow.lstm_batch_builder import DataflowLstmBatchBuilder


app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer(
  "val_graph_count", 10000, "The number of graphs to use in the validation set."
)
app.DEFINE_integer(
  "val_seed", 0xCC, "The seed value for randomly sampling validation graphs.",
)
app.DEFINE_integer(
  "max_data_flow_steps",
  30,
  "If > 0, limit the size of dataflow-annotated graphs used to only those "
  "with data_flow_steps <= --max_data_flow_steps",
)
app.DEFINE_list(
  "train_graph_counts",
  [
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    100000,
    200000,
    300000,
    400000,
    500000,
    1000000,
  ],
  "The list of cumulative training graph counts to evaluate at.",
)
app.DEFINE_boolean(
  "cdfg",
  False,
  "If set, use the CDFG representation for programs. Defaults to ProGraML "
  "representations.",
)
app.DEFINE_integer(
  "max_vocab_size",
  0,
  "If > 0, limit the size of the vocabulary to this number.",
)
app.DEFINE_float(
  "target_vocab_cumfreq", 1.0, "The target cumulative frequency that."
)
app.DEFINE_boolean("test", True, "Whether to test the model after training.")
app.DEFINE_string(
  "run_id",
  None,
  "Optionally specify a name for the run. This must be unique. If not "
  "provided, a run ID is generated using the current time.",
)
app.DEFINE_input_path(
  "model_to_test", None, "The working directory for writing logs", is_dir=True
)
FLAGS = app.FLAGS


def TrainDataflowLSTM(
  path: pathlib.Path, vocab: Dict[str, int], val_seed: int,
) -> pathlib.Path:
  if not path.is_dir():
    raise FileNotFoundError(path)

  # Create the logging directories.
  log_dir, log_relpath = dataflow.CreateLoggingDirectories(
    dataset_root=path,
    model_name="ncc",
    analysis=FLAGS.analysis,
    run_id=FLAGS.run_id,
  )

  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Lstm(vocabulary=vocab, test_only=False, node_y_dimensionality=2,)

  # Read val batches asynchronously
  val_batches = AsyncBatchBuilder(
    batch_builder=DataflowLstmBatchBuilder(
      graph_loader=DataflowGraphLoader(
        path,
        epoch_type=epoch_pb2.VAL,
        analysis=FLAGS.analysis,
        min_graph_count=FLAGS.val_graph_count,
        max_graph_count=FLAGS.val_graph_count,
        data_flow_step_max=FLAGS.max_data_flow_steps,
        logfile=open(log_dir / "graph_loader" / "val.txt", "w"),
        seed=val_seed,
      ),
      vocabulary=vocab,
      padded_sequence_length=model.padded_sequence_length,
      batch_size=model.batch_size,
    ),
  )
  val_batches.start()

  # Cumulative totals for training graph counts at each "epoch".
  train_graph_counts = [int(x) for x in FLAGS.train_graph_counts]
  train_graph_cumsums = np.array(train_graph_counts, dtype=np.int32)
  # The number of training graphs in each "epoch".
  train_graph_counts = train_graph_cumsums - np.concatenate(
    ([0], train_graph_counts[:-1])
  )

  for epoch_step, (train_graph_cumsum, train_graph_count) in enumerate(
    zip(train_graph_cumsums, train_graph_counts), start=1
  ):
    start_time = time.time()

    train_batches = ppar.ThreadedIterator(
      DataflowLstmBatchBuilder(
        DataflowGraphLoader(
          path,
          epoch_type=epoch_pb2.TRAIN,
          analysis=FLAGS.analysis,
          min_graph_count=train_graph_count,
          max_graph_count=train_graph_count,
          data_flow_step_max=FLAGS.max_data_flow_steps,
          logfile=open(
            log_dir / "graph_loader" / f"{epoch_step:03d}.train.txt", "w"
          ),
        ),
        vocabulary=vocab,
        padded_sequence_length=model.padded_sequence_length,
        batch_size=model.batch_size,
      ),
      max_queue_size=100,
    )

    train_results = model.RunBatches(
      epoch_pb2.TRAIN,
      train_batches,
      log_prefix=f"Train to {humanize.Commas(train_graph_cumsum)} graphs",
      total_graph_count=train_graph_count,
    )

    # During validation, wait for the batch builder to finish and then
    # iterate over those.
    val_batches.join()
    val_results = model.RunBatches(
      epoch_pb2.VAL,
      val_batches.batches,
      log_prefix=f"Val at {humanize.Commas(train_graph_cumsum)} graphs",
      total_graph_count=FLAGS.val_graph_count,
    )

    # Write the epoch to file as an epoch list.
    # cat *.EpochList.pbtxt > epochs.pbtxt
    epoch = epoch_pb2.EpochList(
      epoch=[
        epoch_pb2.Epoch(
          walltime_seconds=time.time() - start_time,
          epoch_num=epoch_step,
          train_results=train_results,
          val_results=val_results,
        )
      ]
    )
    print(epoch, end="")
    epoch_relpath = f"epochs/{epoch_step:03d}.EpochList.pbtxt"
    checkpoint_relpath = f"checkpoints/{epoch_step:03d}.Checkpoint.pb"
    pbutil.ToFile(epoch, log_dir / epoch_relpath)
    app.Log(1, "Wrote %s/%s", log_relpath, epoch_relpath)
    pbutil.ToFile(model.SaveCheckpoint(), log_dir / checkpoint_relpath)
    app.Log(1, "Wrote %s/%s", log_relpath, checkpoint_relpath)
  return log_dir


def TestDataflowLSTM(
  path: pathlib.Path,
  log_dir: pathlib.Path,
  vocab: Dict[str, int],
  use_cdfg: bool,
):
  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Create the logging directories.
  assert (log_dir / "epochs").is_dir()
  assert (log_dir / "checkpoints").is_dir()
  assert (log_dir / "graph_loader").is_dir()

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Lstm(vocabulary=vocab, test_only=True, node_y_dimensionality=2,)
  restored_epoch, checkpoint = dataflow.SelectCheckpoint(log_dir)
  model.RestoreCheckpoint(checkpoint)

  batches = DataflowLstmBatchBuilder(
    graph_loader=DataflowGraphLoader(
      path,
      epoch_type=epoch_pb2.TEST,
      analysis=FLAGS.analysis,
      data_flow_step_max=FLAGS.max_data_flow_steps,
      logfile=open(log_dir / "graph_loader" / "test.txt", "w"),
      use_cdfg=use_cdfg,
    ),
    vocabulary=vocab,
    padded_sequence_length=model.padded_sequence_length,
    batch_size=model.batch_size,
  )

  start_time = time.time()
  test_results = model.RunBatches(epoch_pb2.TEST, batches, log_prefix="Test")
  epoch = epoch_pb2.EpochList(
    epoch=[
      epoch_pb2.Epoch(
        walltime_seconds=time.time() - start_time,
        epoch_num=restored_epoch.epoch_num,
        test_results=test_results,
      )
    ]
  )
  print(epoch, end="")

  epoch_path = log_dir / "epochs" / "TEST.EpochList.pbtxt"
  pbutil.ToFile(epoch, epoch_path)
  app.Log(1, "Wrote %s", epoch_path)


def Main():
  """Main entry point."""
  path = pathlib.Path(FLAGS.path)

  with vocabulary.VocabularyZipFile.CreateFromPublishedResults() as inst2vec:
    vocab = inst2vec.dictionary

  if FLAGS.model_to_test:
    log_dir = FLAGS.model_to_test
  else:
    log_dir = TrainDataflowLSTM(
      path=path, vocab=vocab, val_seed=FLAGS.val_seed,
    )

  if FLAGS.test:
    TestDataflowLSTM(
      path=path, vocab=vocab, log_dir=log_dir,
    )


if __name__ == "__main__":
  app.Run(Main)
