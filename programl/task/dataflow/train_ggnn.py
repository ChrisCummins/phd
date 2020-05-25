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
"""Train a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib

from labm8.py import app
from programl.task.dataflow import ggnn
from programl.task.dataflow import vocabulary

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
  "batch_size",
  50000,
  "The number of nodes in a graph. "
  "On our system, we observed that a batch size of 50,000 nodes requires "
  "about 5.2GB of GPU VRAM.",
)
app.DEFINE_boolean(
  "limit_max_data_flow_steps",
  True,
  "If set, limit the size of dataflow-annotated graphs used to only those with "
  "data_flow_steps <= message_passing_step_count",
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
app.DEFINE_boolean(
  "cprofile", False, "Whether to profile the run of the model."
)
app.DEFINE_string(
  "run_id",
  None,
  "Optionally specify a name for the run. This must be unique. If not "
  "provided, a run ID is generated using the current time. If --restore_from "
  "is set, the ID of the restored run is used and this flag has no effect.",
)
app.DEFINE_input_path(
  "restore_from", None, "The working directory for writing logs", is_dir=True
)
app.DEFINE_boolean("test", True, "Whether to test the model after training.")
FLAGS = app.FLAGS


def Main():
  """Main entry point."""

  if FLAGS.cprofile:
    assert not FLAGS.test, "don't run --test when you --cprofile the code."
    import cProfile

    profile = cProfile.Profile()
    # profile.runctx("sleeep()", globals(), locals())

    profstr = """log_dir = dataflow.TrainDataflowGGNN(
    path=pathlib.Path(FLAGS.path),
    analysis=FLAGS.analysis,
    limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
    train_graph_counts=[int(x) for x in FLAGS.train_graph_counts],
    val_graph_count=FLAGS.val_graph_count,
    val_seed=FLAGS.val_seed,
    batch_size=FLAGS.batch_size,
    use_cdfg=FLAGS.cdfg,
    max_vocab_size=FLAGS.max_vocab_size,
    target_vocab_cumfreq=FLAGS.target_vocab_cumfreq,
    run_id=FLAGS.run_id,
  )"""

    profile.runctx(profstr, globals(), locals())
    profile.dump_stats("/home/zacharias/profiling/out.profile")

    print("printing results")
    import pstats

    p = pstats.Stats(profile)
    p.sort_stats("tottime").print_stats(50)
    return

  path = pathlib.Path(FLAGS.path)

  vocab = vocabulary.LoadVocabulary(
    path,
    model_name="cdfg" if FLAGS.cdfg else "programl",
    max_items=FLAGS.max_vocab_size,
    target_cumfreq=FLAGS.target_vocab_cumfreq,
  )

  # CDFG doesn't use positional embeddings.
  if FLAGS.cdfg:
    FLAGS.use_position_embeddings = False

  log_dir = ggnn.TrainDataflowGGNN(
    path=path,
    analysis=FLAGS.analysis,
    vocab=vocab,
    limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
    train_graph_counts=[int(x) for x in FLAGS.train_graph_counts],
    val_graph_count=FLAGS.val_graph_count,
    val_seed=FLAGS.val_seed,
    batch_size=FLAGS.batch_size,
    use_cdfg=FLAGS.cdfg,
    run_id=FLAGS.run_id,
    restore_from=FLAGS.restore_from,
  )

  if FLAGS.test:
    ggnn.TestDataflowGGNN(
      path=path,
      log_dir=log_dir,
      analysis=FLAGS.analysis,
      vocab=vocab,
      limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
      batch_size=FLAGS.batch_size,
      use_cdfg=FLAGS.cdfg,
    )


if __name__ == "__main__":
  app.Run(Main)
