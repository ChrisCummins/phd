"""Create and insert dead code into an OpenCL kernel.

***WARNING***

  This is NOT suitable for general purpose code manipulation. The functions
  here assume that the source code has been formatted according the CLgen
  OpenCL code rewriter. This allows it to make several simplifying assumptions
  about the code, e.g. that the indentation is 2 spaces, that kernel signatures
  have a fixed formatting, etc. If you try this on unformatted code, it will
  likely break (silently).
"""
import re
import typing

import numpy as np

from deeplearning.clgen.preprocessors import opencl
from experimental.compilers.reachability import control_flow_graph_generator
from gpu.cldrive.legacy import args
from labm8 import app
from labm8 import fmt

FLAGS = app.FLAGS


class OpenClFunction(object):
  """Representation of an OpenCL function.

  Can be either: a kernel definition, a function definition, or a function
  declaration.
  """

  def __init__(self, src: str, is_kernel: bool = True):
    self.src = src
    self.is_kernel = is_kernel

  def SetFunctionName(self, new_name: str):
    """Set function name to a new name.

    Args:
      new_name: The new name to set.

    Raises:
      ValueError: If the name could not be set.
    """
    if self.is_kernel:
      self.src, num_replacements = re.subn(r'^kernel void ([A-Z]+)\(',
                                           f'kernel void {new_name}(', self.src)
    else:
      self.src, num_replacements = re.subn(r'^void ([A-Z]+)\(',
                                           f'void {new_name}(', self.src)
    if num_replacements != 1:
      raise ValueError(f"{num_replacements} substitutions made when trying to "
                       f"set function name to '{new_name}'")

  def InsertBlockIntoKernel(self, rand: np.random.RandomState,
                            block_to_insert: str) -> None:
    """Insert a code block at a random position in the kernel.

    Args:
      rand: A random seed.
      block_to_insert: The code block to insert, as a string.
    """
    if not self.is_kernel:
      raise TypeError("Cannot insert block into non-kernel.")

    lines = self.src.split('\n')
    if len(lines) < 2:
      raise ValueError("OpenCL kernel is less than two lines long.")
    # Try and find a point to
    indices = list(range(1, len(lines)))
    rand.shuffle(indices)
    for insertion_line_idx in indices:
      previous_line = lines[insertion_line_idx - 1]
      if previous_line[-1] == ';' or previous_line[-1] == '{':
        # The previous line was either a statement or the start of a new block: we
        # can insert the block here.
        break
      else:
        app.Debug(
            'Previous line "%s" not valid as a code block insertion '
            'point', previous_line)
    else:
      raise ValueError(
          f"Failed to find a position to insert block in function '{self.src}'")

    pre = lines[:insertion_line_idx]
    post = lines[insertion_line_idx:]

    indendation_at_point_of_insertion = 0
    for c in pre[-1]:
      if c == ' ':
        indendation_at_point_of_insertion += 1
      else:
        break
    else:
      raise ValueError(f"Line contains nothing but whitespace: '{pre[-1]}'")

    if previous_line[-1] == '{':
      # Inserting block at the start of a new block, increase indentation.
      indendation_at_point_of_insertion += 2

    if indendation_at_point_of_insertion < 2:
      raise ValueError("Line has insufficient indentation "
                       f"({indendation_at_point_of_insertion}): '{pre[-1]}'")

    block = fmt.Indent(indendation_at_point_of_insertion, block_to_insert)

    self.src = '\n'.join(pre + [block] + post)


def GetKernelArguments(kernel: str):
  try:
    # Extract everything up to the function body, and use an empty function
    # body for parsing. This means that errors that are in the function body
    # will not cause this to fail. E.g. given kernel:
    #
    #   kernel void A(const int a, global float* b) {
    #     b[0] += a;
    #   }
    #
    # This will parse:
    #
    #   kernel void A(const int a, global float* b) {}
    kernel_declaration = kernel[:kernel.index('{')] + '{}'
    return args.GetKernelArguments(kernel_declaration)
  except ValueError as e:
    app.Error("Failure processing kernel: '%s'", kernel)
    raise e


def KernelToFunctionDeclaration(kernel: str) -> OpenClFunction:
  """Build a function declaration for an OpenCL kernel.

  Args:
    kernel: The kernel function to declare.

  Returns:
    A single line function declaration.

  Raises:
    ValueError: If kernel is invalid.
  """
  match = re.match(r'^(kernel )?void ([A-Z]+)\(', kernel)
  if not match:
    raise ValueError("Not a valid OpenCL function")
  name = match.group(2)
  args_string = ', '.join(str(a) for a in GetKernelArguments(kernel))
  return OpenClFunction(f'void {name}({args_string});', is_kernel=False)


def KernelArgumentToVariableDeclaration(arg: args.KernelArg) -> str:
  s = []
  for qual in arg.quals:
    if (qual != "global" and qual != "local" and qual != "constant" and
        qual != 'const'):
      s.append(qual)
  s.append(arg.typename)
  if arg.is_pointer:
    s.append("*")
  if arg.name:
    s.append(arg.name)
  return " ".join(s) + ";"


def KernelToDeadCodeBlock(kernel: str) -> str:
  # Convert arguments to variable declarations.
  declarations = [
      KernelArgumentToVariableDeclaration(a) for a in GetKernelArguments(kernel)
  ]
  # The block header is the list of argument variable declarations.
  header = '\n'.join(fmt.IndentList(2, declarations))
  # The block body is the kernel body. The kernel body is already indented, no
  # need to indent further.
  body = '\n'.join(kernel.split('\n')[1:-1])
  # Wrap block in `if (0) { ... }` conditional.
  return f"if (0) {{\n{header}\n{body}\n}}"


def KernelToFunction(kernel: str) -> OpenClFunction:
  if not kernel.startswith('kernel void '):
    raise ValueError("Invalid kernel")
  else:
    return OpenClFunction(kernel[len('kernel '):], is_kernel=False)


class OpenClDeadcodeInserter(object):
  """A dead code OpenCL source mutator."""

  def __init__(self, rand: np.random.RandomState, kernel: str,
               candidate_kernels: typing.List[str]):
    """Constructor.

    Args:
      rand: A random number state.
      kernel: An OpenCL kernel string.
      candidate_kernels: A list of OpenCL kernel strings to use for deadcode
        insertion.
    """

    def _PreprocessKernel(src: str) -> str:
      """Format a kernel for use and check that it meets requirements."""
      src = opencl.StripDoubleUnderscorePrefixes(src.strip())
      if not src.startswith("kernel void "):
        raise ValueError("Invalid kernel")
      # Strip trailing whitespace, and exclude blank lines.
      return '\n'.join(ln.rstrip() for ln in src.split('\n') if ln.rstrip())

    self._rand = rand

    # A list of code blocks, where each code block is a function definition or
    # declaration.
    self._functions = [
        OpenClFunction(_PreprocessKernel(kernel), is_kernel=True)
    ]

    if not len(candidate_kernels):
      raise ValueError("Must have one or more candidate kernels.")

    self._candidates = [_PreprocessKernel(k) for k in candidate_kernels]

  @property
  def opencl_source(self) -> str:
    """Serialize the mutated source to a string."""
    # Rename the functions.
    gen = control_flow_graph_generator.UniqueNameSequence(base_char='A')
    [cb.SetFunctionName(next(gen)) for cb in self._functions]

    return '\n\n'.join([cb.src for cb in self._functions])

  def Mutate(self) -> None:
    """Run a random mutation."""
    mutator = self._rand.choice([
        self.PrependUnusedFunction, self.AppendUnusedFunction,
        self.PrependUnusedFunctionDeclaration,
        self.AppendUnusedFunctionDeclaration, self.InsertBlockIntoKernel
    ])
    mutator()

  def PrependUnusedFunctionDeclaration(self) -> None:
    # Select a random function to declare.
    to_prepend = self._rand.choice(self._candidates)
    self._functions = [KernelToFunctionDeclaration(to_prepend)
                      ] + self._functions

  def AppendUnusedFunctionDeclaration(self) -> None:
    # Select a random function to declare.
    to_append = self._rand.choice(self._candidates)
    self._functions.append(KernelToFunctionDeclaration(to_append))

  def PrependUnusedFunction(self) -> None:
    to_prepend = self._rand.choice(self._candidates)
    self._functions = [KernelToFunction(to_prepend)] + self._functions

  def AppendUnusedFunction(self) -> None:
    to_append = self._rand.choice(self._candidates)
    self._functions.append(KernelToFunction(to_append))

  def InsertBlockIntoKernel(self) -> None:
    to_modify = self._rand.choice([f for f in self._functions if f.is_kernel])
    to_insert = self._rand.choice(self._candidates)
    to_modify.InsertBlockIntoKernel(self._rand,
                                    KernelToDeadCodeBlock(to_insert))


def GenerateDeadcodeMutations(
    kernels: typing.Iterator[str],
    rand: np.random.RandomState,
    num_permutations_of_kernel: int = 5,
    num_mutations_per_kernel: typing.Tuple[int, int] = (1, 5)) -> \
    typing.Iterator[str]:
  """Generate dead code mutations for a set of kernels.

  Args:
    rand: A random seed.
    kernels: The OpenCL kernels to mutate.
    num_permutations_of_kernel: The number of permutations of each kernel to
      generate.
    num_mutations_per_kernel: The minimum and maximum number of mutations to
      apply to each generated kernel.
  """
  for kernel in kernels:
    for _ in range(num_permutations_of_kernel):
      # Apply random mutations to kernel and yield.
      rand_ = np.random.RandomState(rand.randint(0, int(1e9)))

      # Use all kernels (including the current one we're mutating) as candidates
      # for mutation.
      dci = OpenClDeadcodeInserter(rand_, kernel, candidate_kernels=kernels)

      # RandomState.randint() is in range [low,high), hence add one to max to
      # make it inclusive.
      num_mutations = rand.randint(num_mutations_per_kernel[0],
                                   num_mutations_per_kernel[1] + 1)

      for _ in range(num_mutations):
        dci.Mutate()
      yield dci.opencl_source
