"""Run the dataflow models.

Usage:

    $ bazel run //deeplearning/ml4pl/experiments/dataflow:run_models -- \
        --db_stem='sqlite:////tmp/programl/db/'
        --dataset=reachability,domtree \
        --model=zero_r,lstm_ir,lstm_inst2vec,ggnn \
        --tag_suffix=v1
"""
import time

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.ggnn import ggnn
from deeplearning.ml4pl.models.lstm import lstm
from deeplearning.ml4pl.models.zero_r import zero_r
from labm8.py import app
from labm8.py.internal import flags_parsers

FLAGS = app.FLAGS

app.DEFINE_list(
  "dataset",
  ["reachability", "domtree", "liveness"],
  "The name of the dataset to evaluate. One of {amd,nvidia}.",
)
app.DEFINE_list(
  "model",
  ["zero_r", "lstm_ir", "lstm_inst2vec", "ggnn"],
  "The names of the models to evaluate.",
)
app.DEFINE_string(
  "tag_suffix",
  f"v{time.strftime('%y-%m-%dT%H:%M:%S')}",
  "The tag suffix to use for runs.",
)
app.DEFINE_string(
  "db_stem",
  "file:///var/phd/db/cc1.mysql?programl",
  "The stem for database names.",
)


def Main():
  """Main entry point."""
  db_stem = FLAGS.db_stem
  models = FLAGS.model
  tag_suffix = FLAGS.tag_suffix
  datasets = FLAGS.dataset

  # Set model and dataset-invariant flags.
  FLAGS.log_db = flags_parsers.DatabaseFlag(
    log_database.Database,
    f"{db_stem}_dataflow_logs",
    must_exist=False,  # , must_exist=True
  )
  FLAGS.ir_db = flags_parsers.DatabaseFlag(
    ir_database.Database, f"{db_stem}_ir", must_exist=True
  )
  FLAGS.test_on = "improvement_and_last"
  FLAGS.max_train_per_epoch = 5000
  FLAGS.max_val_per_epoch = 1000

  for dataset in datasets:
    graph_db = graph_tuple_database.Database(
      f"{db_stem}_{dataset}", must_exist=True
    )
    FLAGS.graph_db = flags_parsers.DatabaseFlag(
      graph_tuple_database.Database, graph_db.url, must_exist=True,
    )

    # Use binary prec/rec/f1 scores for binary node classification tasks.
    if graph_db.node_y_dimensionality == 3:
      # alias_sets uses 3-D node labels:
      FLAGS.batch_scores_averaging_method = "weighted"
    elif graph_db.node_y_dimensionality == 2:
      # Binary node classification.
      FLAGS.batch_scores_averaging_method = "binary"
    else:
      raise ValueError(
        f"Unknown node dimensionality: {graph_db.node_y_dimensionality}"
      )

    # Liveness is identifier-based, all others are statement-based.
    if dataset == "liveness":
      FLAGS.nodes = flags_parsers.EnumFlag(
        lstm.NodeEncoder, lstm.NodeEncoder.IDENTIFIER
      )
    else:
      FLAGS.nodes = flags_parsers.EnumFlag(
        lstm.NodeEncoder, lstm.NodeEncoder.STATEMENT
      )

    for model in models:
      FLAGS.tag = f"{dataset}_{model}_{tag_suffix}"

      if model == "zero_r":
        FLAGS.epoch_count = 1
        FLAGS.graph_reader_order = "in_order"
        run.Run(zero_r.ZeroR)
      elif model == "lstm_ir":
        FLAGS.epoch_count = 50
        FLAGS.ir2seq = flags_parsers.EnumFlag(
          lstm.Ir2SeqType, lstm.Ir2SeqType.LLVM
        )
        FLAGS.graph_reader_order = "batch_random"
        FLAGS.padded_sequence_length = 15000
        FLAGS.batch_size = 64
        run.Run(lstm.GraphLstm)
      elif model == "lstm_inst2vec":
        FLAGS.epoch_count = 50
        FLAGS.ir2seq = flags_parsers.EnumFlag(
          lstm.Ir2SeqType, lstm.Ir2SeqType.INST2VEC
        )
        FLAGS.graph_reader_order = "batch_random"
        FLAGS.padded_sequence_length = 15000
        FLAGS.batch_size = 64
        run.Run(lstm.GraphLstm)
      elif model == "ggnn":
        FLAGS.layer_timesteps = ["30"]
        FLAGS.graph_batch_node_count = 15000
        FLAGS.graph_reader_order = "global_random"
        FLAGS.epoch_count = 300
        run.Run(ggnn.Ggnn)
      else:
        raise app.UsageError(f"Unknown model: {model}")


if __name__ == "__main__":
  app.Run(Main)
