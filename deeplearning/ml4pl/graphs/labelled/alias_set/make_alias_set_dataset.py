"""This module prepares alias set datasets."""
import pathlib
import sys
import tempfile
import traceback
import typing

from labm8 import app
from labm8 import fs

from compilers.llvm import opt
from compilers.llvm import opt_util
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import \
  make_data_flow_analysis_dataset as make_dataset
from deeplearning.ml4pl.graphs.labelled.alias_set import alias_set
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util

FLAGS = app.FLAGS


def RunOpt(bytecode: str):
  cfg_dots = []

  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-cfg pass writes files
    # to the current working directory.
    with fs.chdir(output_dir):
      process = opt.Exec(
          ['-dot-cfg', '-basicaa', '-print-alias-sets', '-disable-output'],
          stdin=bytecode,
          universal_newlines=True,
          log=False)

      # Propagate failures from opt as OptExceptions.
      if process.returncode:
        raise opt.OptException(returncode=process.returncode,
                               stderr=process.stderr)

      for file in output_dir.iterdir():
        # Opt pass prints the name of the dot files it generates, e.g.:
        #
        #     $ opt -dot-cfg < foo.ll
        #     WARNING: You're attempting to print out a bitcode file.
        #     This is inadvisable as it may cause display problems. If
        #     you REALLY want to taste LLVM bitcode first-hand, you
        #     can force output with the `-f' option.
        #
        #     Writing 'cfg.DoSomething.dot'...
        #     Writing 'cfg.main.dot'...
        if f"Writing '{file.name}'..." not in stderr:
          raise OSError(f"Could not find reference to file '{file.name}' in "
                        f'opt stderr:\n{process.stderr}')
        with open(file) as f:
          cfg_dots = f.read()

  lines = process.stdout.split('\n')
  lines = [l for l in lines if not l.startswith("Writing '")]
  alias_sets = opt_util.ParseAliasSetsOutput(lines)
  if len(alias_sets) != len(cfg_dots):
    raise ValueError(f"{len(cfg_dots)} CFGs produced, but only "
                     f"{len(alias_sets)} alias sets extracted")
  return cfg_dots, alias_sets


def _ProcessBytecodeJob(
    job: make_dataset.BytecodeJob) -> typing.List[graph_database.GraphMeta]:
  """

  Args:
    packed_args: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.

  Returns:
    A list of reachability-annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id = job
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow='nodes_and_edges',
      preprocess_text=False,
      discard_unknown_statements=False,
  )

  try:
    cfg_dots, alias_sets = RunOpt(bytecode)
  except Exception as e:
    _, _, tb = sys.exc_info()
    tb = traceback.extract_tb(tb, 2)
    filename, line_number, function_name, *_ = tb[-1]
    filename = pathlib.Path(filename).name
    app.Error(
        'Failed to create meta graphs from bytecode '
        '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
        function_name,
        type(e).__name__)
    return []

  false, true = make_dataset.GetFalseTrueType()

  graph_metas = []
  for cfg_dot in cfg_dots:
    try:
      cfg = llvm_util.ControlFlowGraphFromDotSource(cfg_dot)
      if cfg.name not in alias_sets:
        raise ValueError(
            f"CFG with name `{cfg.name}` has no entry in alias sets with keys "
            f"`{list(alias_sets.keys())}`")
      fn_alias_set = alias_sets[cfg.name]

      pointer_counts = [len(a.pointers) for a in fn_alias_set]
      if max(pointer_counts) == 1:
        # Nothing interesting here.
        app.Log(1, 'Skipping CFG with no alias sets with more than one pointer')
        continue

      graph = builder.BuildFromControlFlowGraph(cfg)
      graph.source_name = source_name
      graph.relpath = relpath
      graph.bytecode_id = str(bytecode_id)
      graph.language = language

      annotated_graphs = alias_set.MakeAliasSetGraphs(
          graph,
          alias_sets,
          FLAGS.max_instances_per_graph,
          false=false,
          true=true)
      graph_metas += [
          graph_database.GraphMeta.CreateFromNetworkX(annotated_graph)
          for annotated_graph in annotated_graphs
      ]
    except Exception as e:
      _, _, tb = sys.exc_info()
      tb = traceback.extract_tb(tb, 2)
      filename, line_number, function_name, *_ = tb[-1]
      filename = pathlib.Path(filename).name
      app.Error(
          'Failed to create meta graphs from bytecode '
          '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
          function_name,
          type(e).__name__)

  return graph_metas


class AliasSetExporter(make_dataset.BytecodeExporter):
  """Export from LLVM bytecodes."""

  def GetProcessJobFunction(self):
    return _ProcessBytecodeJob


def main():
  """Main entry point."""
  if not FLAGS.bytecode_db:
    raise app.UsageError('--db required')
  make_dataset.Run(AliasSetExporter)


if __name__ == '__main__':
  app.Run(main)
