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
"""Import bytecodes from from Linux sources."""
import multiprocessing
import pathlib
import typing

import pandas as pd
import progressbar
import pyparsing

from compilers.llvm import clang
from compilers.llvm import opt
from compilers.llvm import opt_util
from datasets.linux import linux
from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs.llvm2graph.legacy.cfg import (
  control_flow_graph as cfg,
)
from deeplearning.ml4pl.graphs.llvm2graph.legacy.cfg import llvm_util
from labm8.py import app
from labm8.py import decorators


FLAGS = app.FLAGS

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "Path of database to populate.",
)


def BytecodeFromLinuxSrc(path: pathlib.Path, optimization_level: str) -> str:
  """Create bytecode from a Linux source file.

  Args:
    path: The path of the source file.
    optimization_level: The clang optimization level to use, one of
        {-O0,-O1,-O2,-O3,-Ofast,-Os,-Oz}.

  Returns:
    The bytecode as a string.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  root = linux.LinuxSourcesDataset().src_tree_root
  genroot = linux.LinuxSourcesDataset().generated_hdrs_root
  # A subset of the arguments found by running `make V=1` in the linux
  # build and grabbing a random C compile target.
  # The build I took this from: Wp,-MD,arch/x86/kernel/.asm-offsets.s.d  -nostdinc -isystem /usr/lib/gcc/x86_64-linux-gnu/5/include -I./arch/x86/include -I./arch/x86/include/generated  -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/kconfig.h -include ./include/linux/compiler_types.h -D__KERNEL__ -Wall -Wundef -Wstrict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common -fshort-wchar -Werror-implicit-function-declaration -Wno-format-security -std=gnu89 -fno-PIE -DCC_HAVE_ASM_GOTO -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -funit-at-a-time -DCONFIG_X86_X32_ABI -DCONFIG_AS_CFI=1 -DCONFIG_AS_CFI_SIGNAL_FRAME=1 -DCONFIG_AS_CFI_SECTIONS=1 -DCONFIG_AS_FXSAVEQ=1 -DCONFIG_AS_SSSE3=1 -DCONFIG_AS_CRC32=1 -DCONFIG_AS_AVX=1 -DCONFIG_AS_AVX2=1 -DCONFIG_AS_AVX512=1 -DCONFIG_AS_SHA1_NI=1 -DCONFIG_AS_SHA256_NI=1 -pipe -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -DRETPOLINE -fno-delete-null-pointer-checks -O2 --param=allow-store-data-races=0 -Wframe-larger-than=1024 -fstack-protector-strong -Wno-unused-but-set-variable -fno-var-tracking-assignments -g -gdwarf-4 -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -Wdeclaration-after-statement -Wno-pointer-sign -fno-strict-overflow -fno-merge-all-constants -fmerge-constants -fno-stack-check -fconserve-stack -Werror=implicit-int -Werror=strict-prototypes -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init    -DKBUILD_BASENAME='"asm_offsets"' -DKBUILD_MODNAME='"asm_offsets"'  -fverbose-asm -S -o arch/x86/kernel/asm-offsets.s arch/x86/kernel/asm-offsets.c
  clang_args = [
    "-S",
    "-emit-llvm",
    "-o",
    "-",
    clang.ValidateOptimizationLevel(optimization_level),
    "-Wno-everything",  # No warnings please.
    "-I",
    str(root / "arch/x86/include"),
    "-I",
    str(genroot / "arch/x86/include/generated"),
    "-I",
    str(root / "include"),
    "-I",
    str(root / "arch/x86/include/uapi"),
    "-I",
    str(genroot / "arch/x86/include/generated/uapi"),
    "-I",
    str(root / "include/uapi"),
    "-I",
    str(genroot / "include/generated/uapi"),
    "-I",
    str(genroot / "arch/x86/include"),
    "-I",
    str(genroot / "arch/x86/include/generated"),
    "-I",
    str(genroot / "arch/x86/include/generated/uapi"),
    "-I",
    str(genroot / "include"),
    "-I",
    str(genroot / "include/generated"),
    "-include",
    str(genroot / "include/linux/kconfig.h"),
    "-include",
    str(genroot / "include/linux/compiler_types.h"),
    "-D__KERNEL__",
    "-m64",
    "-DCONFIG_X86_X32_ABI",
    "-DCONFIG_AS_CFI=1",
    "-DCONFIG_AS_CFI_SIGNAL_FRAME=1",
    "-DCONFIG_AS_CFI_SECTIONS=1",
    "-DCONFIG_AS_FXSAVEQ=1",
    "-DCONFIG_AS_SSSE3=1",
    "-DCONFIG_AS_CRC32=1",
    "-DCONFIG_AS_AVX=1",
    "-DCONFIG_AS_AVX2=1",
    "-DCONFIG_AS_AVX512=1",
    "-DCONFIG_AS_SHA1_NI=1",
    "-DCONFIG_AS_SHA256_NI=1",
    "-pipe",
    "-DRETPOLINE",
    "-DCC_USING_FENTRY",
    "-DKBUILD_BASENAME='\"asm_offsets\"'",
    "-DKBUILD_MODNAME='\"asm_offsets\"'",
    str(path),
  ]
  process = clang.Exec(clang_args)
  if process.returncode:
    raise clang.ClangException(
      returncode=process.returncode, stderr=process.stderr, command=clang_args
    )
  return process.stdout, clang_args


def TryToCreateControlFlowGraphsFromLinuxSrc(
  path: pathlib.Path,
) -> typing.List[cfg.ControlFlowGraph]:
  """Try to create CFGs from a Linux C source file.

  On failure, an empty list is returned.

  Args:
    path: The path of the source file.

  Returns:
    A list of ControlFlowGraph instances.

  Raises:
    ClangException: If compiling to bytecode fails.
  """
  graphs = []

  try:
    bytecode, _ = BytecodeFromLinuxSrc(path, "-O0")
  except clang.ClangException:
    return graphs

  # Extract a dot sources from the bytecode.
  dot_generator = opt_util.DotControlFlowGraphsFromBytecode(bytecode)
  while True:
    try:
      dot = next(dot_generator)
      # Instantiate a CFG from the dot source.
      graph = llvm_util.ControlFlowGraphFromDotSource(dot)
      graph.ValidateControlFlowGraph(strict=False)
      graphs.append(graph)
    except (
      UnicodeDecodeError,
      cfg.MalformedControlFlowGraphError,
      ValueError,
      opt.OptException,
      pyparsing.ParseException,
    ):
      pass
    except StopIteration:
      break

  return graphs


def CfgDfRowFromControlFlowGraph(
  graph: cfg.ControlFlowGraph,
) -> typing.Dict[str, typing.Any]:
  return {
    "cfg:graph": graph,
    "cfg:block_count": graph.number_of_nodes(),
    "cfg:edge_count": graph.number_of_edges(),
    "cfg:edge_density": graph.edge_density,
    "cfg:is_valid": graph.IsValidControlFlowGraph(strict=False),
    "cfg:is_strict_valid": graph.IsValidControlFlowGraph(strict=True),
  }


def ProcessLinuxSrc(
  path: pathlib.Path,
) -> typing.Optional[typing.Dict[str, typing.Any]]:
  graphs = TryToCreateControlFlowGraphsFromLinuxSrc(path)

  src_root = LinuxSourcesDataset().src_tree_root

  rows = []
  for graph in graphs:
    row = CfgDfRowFromControlFlowGraph(graph)
    row.update(
      {"program:src_relpath": str(path)[len(str(src_root)) + 1 :],}
    )
    rows.append(row)

  return rows


def ProcessLinuxSrcToBytecode(path: pathlib.Path) -> ml4pl_pb2.LlvmBytecode:
  src_root = LinuxSourcesDataset().src_tree_root
  version = LinuxSourcesDataset().version

  try:
    bytecode, cflags = BytecodeFromLinuxSrc(path, "-O0")
    clang_returncode = 0
    error_message = ""
  except clang.ClangException as e:
    bytecode = ""
    cflags = e.command
    clang_returncode = e.returncode
    error_message = e.stderr

  return ml4pl_pb2.LlvmBytecode(
    source_name=f"linux-{version}",
    relpath=str(path)[len(str(src_root)) + 1 :],
    lang="C",
    cflags=" ".join(cflags),
    bytecode=bytecode,
    clang_returncode=clang_returncode,
    error_message=error_message,
  )


class LinuxSourcesDataset(linux.LinuxSourcesDataset):
  """Control flow graphs from a subset of the Linux source tree.

  The returned DataFrame has the following schema:

    program:src_relpath (str): The path of the source file within the linux
      source tree.
    cfg:graph (ControlFlowGraph): A control flow graph instance.
    cfg:block_count (int): The number of basic blocks in the CFG.
    cfg:edge_count (int): The number of edges in the CFG.
    cfg:edge_density (float): Number of edges / possible edges, in range [0,1].
    cfg:is_valid (bool): Whether the CFG is valid.
    cfg:is_strict_valid (bool): Whether the CFG is valid when strict.
  """

  def PopulateBytecodeTable(
    self, db: bytecode_database.Database, commit_every: int = 1000
  ):
    bar = progressbar.ProgressBar()
    bar.max_value = len(self.all_srcs)

    # Process each row of the table in parallel.
    pool = multiprocessing.Pool()
    with db.Session(commit=True) as s:
      for i, proto in enumerate(
        pool.imap_unordered(ProcessLinuxSrcToBytecode, self.all_srcs)
      ):
        bar.update(i)
        s.add(
          bytecode_database.LlvmBytecode(
            **bytecode_database.LlvmBytecode.FromProto(proto)
          )
        )
        if not (i % commit_every):
          s.commit()

  @decorators.memoized_property
  def cfgs_df(self) -> pd.DataFrame:
    # Process each row of the table in parallel.
    pool = multiprocessing.Pool()
    rows = []
    for row_batch in pool.imap_unordered(ProcessLinuxSrc, self.kernel_srcs):
      if row_batch:
        rows += row_batch

    # Create the output table.
    df = pd.DataFrame(
      rows,
      columns=[
        "program:src_relpath",
        "cfg:graph",
        "cfg:block_count",
        "cfg:edge_count",
        "cfg:edge_density",
        "cfg:is_valid",
        "cfg:is_strict_valid",
      ],
    )

    df.set_index(["program:src_relpath",], inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
  db = FLAGS.bytecode_db()
  dataset = OpenClDeviceMappingsDataset()
  dataset.PopulateBytecodeTable(db)


if __name__ == "__main__":
  app.Run(main)
