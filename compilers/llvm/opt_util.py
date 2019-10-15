"""Utility functions """
import pathlib
import subprocess
import tempfile
import typing

from compilers.llvm import llvm
from compilers.llvm import llvm_as
from compilers.llvm import opt
from labm8 import app
from labm8 import fs


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


def DotCallGraphAndControlFlowGraphsFromBytecode(
    bytecode: str
) -> typing.Tuple[str, typing.List[str]]:
  """Create call graph and control flow graphs from an LLVM bytecode file.

  When both a call graph and CFGs are required, calling this function is
  marginally faster than calling DotControlFlowGraphsFromBytecode() and
  DotCallGraphFromBytecode() separately.

  Args:
    bytecode: The LLVM bytecode to create call graph and CFGs from.

  Returns:
    A tuple, where the first element is the call graph dot string, and the
    second element is a list of control flow graph dot strings.

  Raises:
    OptException: In case the opt pass fails.
    UnicodeDecodeError: If generated dotfile can't be read.
  """
  control_flow_graph_dots = []

  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    output_dir = pathlib.Path(d)
    # Change into the output directory, because the -dot-callgraph pass writes
    # to the current working directory.
    with fs.chdir(output_dir):
      # We run with universal_newlines=False because the stdout of opt is the
      # binary bitcode, which we completely ignore (we're only interested in
      # stderr). This means we must encode stdin and decode stderr ourselves.
      process = opt.Exec(['-dot-cfg', '-dot-callgraph'],
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
        if file.name != 'callgraph.dot':
          control_flow_graph_dots.append(fs.Read(file))

      return fs.Read(callgraph), control_flow_graph_dots


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
