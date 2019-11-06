# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions """
import collections
import pathlib
import re
import subprocess
import tempfile
import typing

from labm8 import app
from labm8 import fs

from compilers.llvm import llvm
from compilers.llvm import llvm_as
from compilers.llvm import opt

FLAGS = app.FLAGS


def DotCallGraphFromBytecode(bytecode: str) -> str:
  """Create a call graph from an LLVM bytecode file.

  Args:
    bytecode: The LLVM bytecode to create call graphfrom.

  Returns:
    A dotfile string.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-callgraph pass writes
    # to the current working directory.
    with fs.chdir(output_dir):
      # We run with universal_newlines=False because the stdout of opt is the
      # binary bitcode, which we completely ignore (we're only interested in
      # stderr). This means we must encode stdin and decode stderr ourselves.
      process = opt.Exec(['-dot-callgraph'],
                         stdin=bytecode.encode('utf-8'),
                         universal_newlines=False,
                         log=False)
      stderr = process.stderr.decode('utf-8')

      # Propagate failures from opt as OptExceptions.
      if process.returncode:
        raise opt.OptException(returncode=process.returncode, stderr=stderr)

      callgraph = output_dir / 'callgraph.dot'
      if not callgraph.is_file():
        raise OSError(f"Callgraph dotfile not produced")
      return fs.Read(callgraph)


def DotControlFlowGraphsFromBytecode(bytecode: str) -> typing.Iterator[str]:
  """Create a control flow graph from an LLVM bytecode file.

  Args:
    bytecode: The LLVM bytecode to create CFG dots from.

  Returns:
    An iterator of dotfile strings.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-cfg pass writes files
    # to the current working directory.
    with fs.chdir(output_dir):
      # We run with universal_newlines=False because the stdout of opt is the
      # binary bitcode, which we completely ignore (we're only interested in
      # stderr). This means we must encode stdin and decode stderr ourselves.
      process = opt.Exec(['-dot-cfg'],
                         stdin=bytecode.encode('utf-8'),
                         universal_newlines=False,
                         log=False)
      stderr = process.stderr.decode('utf-8')

      # Propagate failures from opt as OptExceptions.
      if process.returncode:
        raise opt.OptException(returncode=process.returncode, stderr=stderr)

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
          yield f.read()


def DotGraphsFromBytecode(
    bytecode: str,
    opt_args: typing.List[str],
    opt_path: str = None,
    output_pred: typing.Callable[[str], bool] = None
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Obtain dot graphs from an LLVM bytecode file using an opt pass.

  Args:
    bytecode: The LLVM bytecode to create the graphs from.
    opt_args: A list of arguments to the opt tool that generate the graphs.
    opt_path: The path to a custom opt binary. Overrides the default version.
    output_pred: A predicate that receives an output file name, and returns
                 True if it should be collected in the first part of the result tuple,
                 or False if it should be collected in the second tuple element.
                 If None, all outputs are collected to the first element.

  Returns:
    A 2-tuple of lists of graphs as dot strings.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  graph_dots_true = []
  graph_dots_false = []

  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-callgraph pass writes
    # to the current working directory.
    with fs.chdir(output_dir):
      # We run with universal_newlines=False because the stdout of opt is the
      # binary bitcode, which we completely ignore (we're only interested in
      # stderr). This means we must encode stdin and decode stderr ourselves.
      process = opt.Exec(opt_args,
                         stdin=bytecode.encode('utf-8'),
                         universal_newlines=False,
                         log=False,
                         opt=opt_path)
      stderr = process.stderr.decode('utf-8')

      # Propagate failures from opt as OptExceptions.
      if process.returncode:
        raise opt.OptException(returncode=process.returncode, stderr=stderr)

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
        if not output_pred or output_pred(file.name):
          graph_dots_true.append(fs.Read(file))
        else:
          graph_dots_false.append(fs.Read(file))

      return graph_dots_true, graph_dots_false


def DotCallGraphAndControlFlowGraphsFromBytecode(
    bytecode: str, opt_path: str = None) -> typing.Tuple[str, typing.List[str]]:
  """Create call graph and control flow graphs from an LLVM bytecode file.

  When both a call graph and CFGs are required, calling this function is
  marginally faster than calling DotControlFlowGraphsFromBytecode() and
  DotCallGraphFromBytecode() separately.

  Args:
    bytecode: The LLVM bytecode to create call graph and CFGs from.
    opt_path: The path to a custom opt binary. Overrides the default version.

  Returns:
    A tuple, where the first element is the call graph dot string, and the
    second element is a list of control flow graph dot strings.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  control_flow_graph_dots, callgraph_dots = DotGraphsFromBytecode(
    bytecode, ['-dot-cfg', '-dot-callgraph'], opt_path,
    lambda name: name != 'callgraph.dot'
  )

  if len(callgraph_dots) != 1:
    raise OSError(f"Callgraph dotfile not produced")

  callgraph = callgraph_dots[0]

  return callgraph, control_flow_graph_dots


def GetOptArgs(cflags: typing.Optional[typing.List[str]] = None
              ) -> typing.List[typing.List[str]]:
  """Get the arguments passed to opt.

  Args:
    cflags: The cflags passed to clang. Defaults to -O0.

  Returns:
    A list of invocation arguments.
  """
  cflags = cflags or ['-O0']
  p1 = subprocess.Popen([llvm_as.LLVM_AS],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE)
  p2 = subprocess.Popen(
      [opt.OPT, '-disable-output', '-debug-pass=Arguments'] + cflags,
      stdin=p1.stdout,
      stderr=subprocess.PIPE,
      universal_newlines=True)
  _, stderr = p2.communicate()
  if p2.returncode:
    raise llvm.LlvmError(stderr)
  args = []
  for line in stderr.rstrip().split('\n'):
    if not line.startswith('Pass Arguments:'):
      raise llvm.LlvmError(f'Cannot interpret line: {line}')
    line = line[len('Pass Arguments:'):]
    args.append(line.split())
    for arg in args[-1]:
      if not arg[0] == '-':
        raise llvm.LlvmError(f'Cannot interpret clang argument: {arg}')
  return args


class Pointer(typing.NamedTuple):
  """A pointer in an alias set."""
  type: str
  identifier: str
  size: int


class AliasSet(typing.NamedTuple):
  # From https://llvm.org/doxygen/AliasSetTracker_8h_source.html
  #
  #    /// The kind of alias relationship between pointers of the set.
  #    ///
  #    /// These represent conservatively correct alias results between any members
  #    /// of the set. We represent these independently of the values of alias
  #    /// results in order to pack it into a single bit. Lattice goes from
  #    /// MustAlias to MayAlias.
  #    enum AliasLattice {
  #      SetMustAlias = 0, SetMayAlias = 1
  #    };
  type: str  # str, one of {must alias, may alias}
  # From https://llvm.org/doxygen/AliasSetTracker_8h_source.html
  #
  #    /// The kinds of access this alias set models.
  #    ///
  #    /// We keep track of whether this alias set merely refers to the locations of
  #    /// memory (and not any particular access), whether it modifies or references
  #    /// the memory, or whether it does both. The lattice goes from "NoAccess" to
  #    /// either RefAccess or ModAccess, then to ModRefAccess as necessary.
  #    enum AccessLattice {
  #      NoAccess = 0,
  #      RefAccess = 1,
  #      ModAccess = 2,
  #      ModRefAccess = RefAccess | ModAccess
  #    };
  mod_ref: str  # str, one of {Mod,Ref,Mod/Ref}
  pointers: typing.List[Pointer]


def GetAliasSetsByFunction(
    bytecode: str) -> typing.Dict[str, typing.List[AliasSet]]:
  """Get the alias sets of a bytecode.

  Args:
    bytecode: An LLVM bytecode.

  Returns:
    A dictionary of alias sets, keyed by the function name.

  Raises:
    OptException: In case the opt pass fails.
  """
  process = opt.Exec(['-basicaa', '-print-alias-sets', '-disable-output'],
                     stdin=bytecode,
                     universal_newlines=True,
                     log=False)

  # Propagate failures from opt as OptExceptions.
  if process.returncode:
    raise opt.OptException(returncode=process.returncode, stderr=process.stderr)

  return ParseAliasSetsOutput(process.stderr)


def ParseAliasSetsOutput(
    output: str) -> typing.Dict[str, typing.List[AliasSet]]:
  lines = output.split('\n')
  function_alias_sets = {}

  function = None
  alias_sets = None
  for line in lines:
    if line.startswith("Alias sets for function "):
      function = line[len("Alias sets for function '"):-len(":'")]
      function_alias_sets[function] = []
      alias_sets = function_alias_sets[function]
    elif line.startswith("Alias Set Tracker: "):
      if function is None:
        raise ValueError("Unexpected line!")
    elif line.startswith("  AliasSet["):
      if function is None:
        raise ValueError("Unexpected line!")
      line = line[line.index('] ') + 2:]
      alias_set_type = line[:line.index(',')]
      line = line[line.index(',') + 1:]
      mod_ref = line.split()[0]
      line = ' '.join(line.split()[2:])
      pointers = line.split('),')

      alias_set_pointers = []
      for pointer in pointers:
        pointer = pointer[1:]
        if not pointer:
          continue
        try:
          typename, identifier, size = pointer.split()
        except ValueError:
          raise ValueError(f"Failed to parse line '{line}'")
        identifier = identifier[:-1]
        if size.endswith(')'):
          size = size[:-1]
        pointer = Pointer(type=typename, identifier=identifier, size=int(size))
        alias_set_pointers.append(pointer)
      if alias_set_pointers:
        alias_sets.append(
            AliasSet(
                type=alias_set_type,
                mod_ref=mod_ref,
                pointers=alias_set_pointers,
            ))
    elif 'Unknown instructions' in line:
      pass
    elif line.strip():  # Empty line
      raise ValueError(line)
  return function_alias_sets


AnalysisOutput = collections.namedtuple(
    'AnalysisOutput',
    [
        'analysis',  # str, the name of the analysis
        'function',  # typing.Optional[str], function name, or None if global analysis
        'lines',  # typing.List[str], the output of the analysis
    ])


def RunAnalysisPasses(bytecode: str,
                      passes: typing.List[str],
                      opt_path: typing.Optional[pathlib.Path] = None
                     ) -> typing.Iterable[AnalysisOutput]:
  """Run the given opt analysis passes and parse the output.

  TODO(github.com/ChrisCummins/phd/issues/54): this requires using an LLVM
  built with debug+assert flags, which we do not currently have. You must
  therefore use the `opt_path` argument to pass in the path of an opt binary
  with the debug+assert flags enabled. Care should be taken to match the
  version of opt against the one included in this project as the output format
  may have changed between releases. See //:WORKSPACE for current llvm version.

  Args:
    bytecode: The bytecode to analyse.
    passes: A list of passes to run. See:
      <https://llvm.org/docs/Passes.html#analysis-passes> for a list.

  Returns:
    A generator of AnalysisOutput tuples.
  """
  cmd = ['-analyze', '-stats', '-'] + passes
  p = opt.Exec(cmd, stdin=bytecode, opt=opt_path)
  if p.returncode:
    raise opt.OptException('opt analysis failed',
                           returncode=p.returncode,
                           stderr=p.stderr,
                           command=cmd)
  return ParseAnalysisPassesOutput(p.stdout)


def ParseAnalysisPassesOutput(stdout: str) -> typing.Iterable[AnalysisOutput]:
  """Parse the output of opt's analysis passes.

  Args:
    stdout: The opt stdout to parse.

  Returns:
    A generator of AnalysisOutput tuples.
  """
  global_analysis_re = re.compile(r"Printing analysis '(?P<analysis>[^']+)':")
  function_analysis_re = re.compile(
      r"Printing analysis '(?P<analysis>[^']+)' for function '(?P<function>[^']+)':"
  )

  analysis = None
  for line in stdout.split('\n'):
    line = line.strip()
    if not line:
      continue

    match = global_analysis_re.match(line)
    if match:
      if analysis:  # Emit current analysis
        yield analysis
      analysis = AnalysisOutput(analysis=match.group('analysis'),
                                function=None,
                                lines=[])
      continue

    match = function_analysis_re.match(line)
    if match:
      if analysis:  # Emit current analysis
        yield analysis
      analysis = AnalysisOutput(analysis=match.group('analysis'),
                                function=match.group('function'),
                                lines=[])
      continue

    if analysis:
      analysis.lines.append(line)
    else:
      raise ValueError(f"Unable to parse analysis output: `{p.stdout.strip()}`")
