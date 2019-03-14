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
"""A python wrapper around opt, the LLVM optimizer.

opt is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke opt. Note you must
use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:opt [-- <script_args> [-- <opt_args>]]
"""
import functools
import pathlib
import subprocess
import sys
import typing

from compilers.llvm import llvm
from labm8 import app
from labm8 import bazelutil
from labm8 import system

FLAGS = app.FLAGS

app.DEFINE_integer('opt_timeout_seconds', 60,
                   'The maximum number of seconds to allow process to run.')

_LLVM_REPO = 'llvm_linux' if system.is_linux() else 'llvm_mac'

# Path to opt binary.
OPT = bazelutil.DataPath(f'{_LLVM_REPO}/bin/opt')

# The list of LLVM opt transformation passes.
# See: https://llvm.org/docs/Passes.html#transform-passes
TRANSFORM_PASSES = {
    '-aa',
    '-aa-eval',
    '-aarch64-a57-fp-load-balancing',
    '-aarch64-ccmp',
    '-aarch64-collect-loh',
    '-aarch64-condopt',
    '-aarch64-copyelim',
    '-aarch64-dead-defs',
    '-aarch64-expand-pseudo',
    '-aarch64-fix-cortex-a53-835769-pass',
    '-aarch64-ldst-opt',
    '-aarch64-local-dynamic-tls-cleanup',
    '-aarch64-promote-const',
    '-aarch64-simd-scalar',
    '-aarch64-simdinstr-opt',
    '-aarch64-stp-suppress',
    '-adce',
    '-add-discriminators',
    '-alignment-from-assumptions',
    '-alloca-hoisting',
    '-always-inline',
    '-amdgpu-aa',
    '-amdgpu-always-inline',
    '-amdgpu-annotate-kernel-features',
    '-amdgpu-annotate-uniform',
    '-amdgpu-argument-reg-usage-info',
    '-amdgpu-codegenprepare',
    '-amdgpu-inline',
    '-amdgpu-lower-enqueued-block',
    '-amdgpu-lower-intrinsics',
    '-amdgpu-promote-alloca',
    '-amdgpu-rewrite-out-arguments',
    '-amdgpu-simplifylib',
    '-amdgpu-unify-divergent-exit-nodes',
    '-amdgpu-unify-metadata',
    '-amdgpu-usenative',
    '-amode-opt',
    '-argpromotion',
    '-arm-cp-islands',
    '-arm-execution-deps-fix',
    '-arm-ldst-opt',
    '-arm-prera-ldst-opt',
    '-arm-pseudo',
    '-asan',
    '-asan-module',
    '-assumption-cache-tracker',
    '-atomic-expand',
    '-barrier',
    '-basicaa',
    '-basiccg',
    '-bdce',
    '-block-freq',
    '-block-placement',
    '-bool-ret-to-int',
    '-bounds-checking',
    '-branch-folder',
    '-branch-prob',
    '-break-crit-edges',
    '-called-value-propagation',
    '-callsite-splitting',
    '-cfl-anders-aa',
    '-cfl-steens-aa',
    '-check-debugify',
    '-codegenprepare',
    '-collector-metadata',
    '-consthoist',
    '-constmerge',
    '-constprop',
    '-coro-cleanup',
    '-coro-early',
    '-coro-elide',
    '-coro-split',
    '-correlated-propagation',
    '-cost-model',
    '-cross-dso-cfi',
    '-da',
    '-dce',
    '-dead-mi-elimination',
    '-deadargelim',
    '-deadarghaX0r',
    '-debugify',
    '-delinearize',
    '-demanded-bits',
    '-detect-dead-lanes',
    '-dfsan',
    '-die',
    '-div-rem-pairs',
    '-divergence',
    '-domfrontier',
    '-domtree',
    '-dot-callgraph',
    '-dot-cfg',
    '-dot-cfg-only',
    '-dot-dom',
    '-dot-dom-only',
    '-dot-postdom',
    '-dot-postdom-only',
    '-dot-regions',
    '-dot-regions-only',
    '-dot-scops',
    '-dot-scops-only',
    '-dse',
    '-dwarfehprepare',
    '-early-cse',
    '-early-cse-memssa',
    '-early-ifcvt',
    '-edge-bundles',
    '-ee-instrument',
    '-elim-avail-extern',
    '-esan',
    '-expand-isel-pseudos',
    '-expand-reductions',
    '-expandmemcmp',
    '-external-aa',
    '-extract-blocks',
    '-falkor-hwpf-fix',
    '-falkor-hwpf-fix-late',
    '-fentry-insert',
    '-flattencfg',
    '-float2int',
    '-forceattrs',
    '-funclet-layout',
    '-function-import',
    '-functionattrs',
    '-gc-analysis',
    '-gc-lowering',
    '-generic-to-nvvm',
    '-global-merge',
    '-globaldce',
    '-globalopt',
    '-globals-aa',
    '-globalsplit',
    '-greedy',
    '-guard-widening',
    '-gvn',
    '-gvn-hoist',
    '-gvn-sink',
    '-hexagon-cext-opt',
    '-hexagon-early-if',
    '-hexagon-gen-mux',
    '-hexagon-loop-idiom',
    '-hexagon-nvj',
    '-hexagon-packetizer',
    '-hexagon-rdf-opt',
    '-hexagon-vlcr',
    '-hwasan',
    '-hwloops',
    '-indirectbr-expand',
    '-indvars',
    '-infer-address-spaces',
    '-inferattrs',
    '-inline',
    '-insert-gcov-profiling',
    '-instcombine',
    '-instcount',
    '-instnamer',
    '-instrprof',
    '-instruction-select',
    '-instsimplify',
    '-interleaved-access',
    '-internalize',
    '-intervals',
    '-ipconstprop',
    '-ipsccp',
    '-irce',
    '-irtranslator',
    '-isel',
    '-iv-users',
    '-jump-threading',
    '-lazy-block-freq',
    '-lazy-branch-prob',
    '-lazy-machine-block-freq',
    '-lazy-value-info',
    '-lcssa',
    '-lcssa-verification',
    '-legalizer',
    '-libcalls-shrinkwrap',
    '-licm',
    '-lint',
    '-livedebugvalues',
    '-livedebugvars',
    '-liveintervals',
    '-liveregmatrix',
    '-livestacks',
    '-livevars',
    '-load-store-vectorizer',
    '-localizer',
    '-localstackalloc',
    '-loop-accesses',
    '-loop-data-prefetch',
    '-loop-deletion',
    '-loop-distribute',
    '-loop-extract',
    '-loop-extract-single',
    '-loop-idiom',
    '-loop-instsimplify',
    '-loop-interchange',
    '-loop-load-elim',
    '-loop-predication',
    '-loop-reduce',
    '-loop-reroll',
    '-loop-rotate',
    '-loop-simplify',
    '-loop-simplifycfg',
    '-loop-sink',
    '-loop-unroll',
    '-loop-unswitch',
    '-loop-vectorize',
    '-loop-versioning',
    '-loop-versioning-licm',
    '-loops',
    '-lower-expect'
    '-lower-expect',
    '-lower-guard-intrinsic',
    '-loweratomic',
    '-lowerinvoke',
    '-lowerswitch',
    '-lowertypetests',
    '-lrshrink',
    '-machine-block-freq',
    '-machine-branch-prob',
    '-machine-combiner',
    '-machine-cp',
    '-machine-cse',
    '-machine-domfrontier',
    '-machine-loops',
    '-machine-opt-remark-emitter',
    '-machine-scheduler',
    '-machine-sink',
    '-machine-trace-metrics',
    '-machinedomtree',
    '-machinelicm',
    '-machinemoduleinfo',
    '-machinepostdomtree',
    '-mem2reg',
    '-memcpyopt',
    '-memdep',
    '-memoryssa',
    '-mergefunc',
    '-mergeicmps',
    '-mergereturn',
    '-metarenamer',
    '-mldst-motion',
    '-module-debuginfo',
    '-module-summary-analysis',
    '-msan',
    '-name-anon-globals',
    '-nary-reassociate',
    '-newgvn',
    '-nvptx-assign-valid-global-names',
    '-nvptx-lower-aggr-copies',
    '-nvptx-lower-alloca',
    '-nvptx-lower-args',
    '-nvvm-intr-range',
    '-nvvm-reflect',
    '-objc-arc',
    '-objc-arc-aa',
    '-objc-arc-apelim',
    '-objc-arc-contract',
    '-objc-arc-expand',
    '-opt-phis',
    '-opt-remark-emitter',
    '-pa-eval',
    '-packets',
    '-partial-inliner',
    '-partially-inline-libcalls',
    '-patchable-function',
    '-peephole-opt',
    '-pgo-icall-prom',
    '-pgo-instr-gen',
    '-pgo-instr-use',
    '-pgo-memop-opt',
    '-phi-node-elimination',
    '-place-backedge-safepoints-impl',
    '-place-safepoints',
    '-polly-ast',
    '-polly-canonicalize',
    '-polly-cleanup',
    '-polly-codegen',
    '-polly-dce',
    '-polly-delicm',
    '-polly-dependences',
    '-polly-detect',
    '-polly-dump-module',
    '-polly-export-jscop',
    '-polly-flatten-schedule',
    '-polly-function-dependences',
    '-polly-function-scops',
    '-polly-import-jscop',
    '-polly-mse',
    '-polly-opt-isl',
    '-polly-optree',
    '-polly-prepare',
    '-polly-prune-unprofitable',
    '-polly-rewrite-byref-params',
    '-polly-scop-inliner',
    '-polly-scops',
    '-polly-simplify',
    '-polyhedral-info',
    '-post-inline-ee-instrument',
    '-post-RA-sched',
    '-postdomtree',
    '-postrapseudos',
    '-ppc-expand-isel',
    '-ppc-mi-peepholes',
    '-ppc-pre-emit-peephole',
    '-ppc-tls-dynamic-call',
    '-pre-isel-intrinsic-lowering',
    '-print-alias-sets',
    '-print-bb',
    '-print-callgraph',
    '-print-callgraph-sccs',
    '-print-cfg-sccs',
    '-print-dom-info',
    '-print-externalfnconstants',
    '-print-function',
    '-print-lazy-value-info',
    '-print-memdeps',
    '-print-memderefs',
    '-print-memoryssa',
    '-print-module',
    '-print-predicateinfo',
    '-processimpdefs',
    '-profile-summary-info',
    '-prologepilog',
    '-prune-eh',
    '-r600-expand-special-instrs',
    '-r600cf',
    '-r600mergeclause',
    '-reassociate',
    '-reg2mem',
    '-regbankselect',
    '-regions',
    '-rename-independent-subregs',
    '-rewrite-statepoints-for-gc',
    '-rewrite-symbols',
    '-rpo-functionattrs',
    '-safe-stack',
    '-sample-profile',
    '-sancov',
    '-scalar-evolution',
    '-scalarize-masked-mem-intrin',
    '-scalarizer',
    '-sccp',
    '-scev-aa',
    '-scoped-noalias',
    '-separate-const-offset-from-gep',
    '-shadow-stack-gc-lowering',
    '-shrink-wrap',
    '-si-annotate-control-flow',
    '-si-debugger-insert-nops',
    '-si-fix-sgpr-copies',
    '-si-fix-vgpr-copies',
    '-si-fix-wwm-liveness',
    '-si-fold-operands',
    '-si-i1-copies',
    '-si-insert-skips',
    '-si-insert-waitcnts',
    '-si-insert-waits',
    '-si-load-store-opt',
    '-si-lower-control-flow',
    '-si-memory-legalizer',
    '-si-optimize-exec-masking',
    '-si-optimize-exec-masking-pre-ra',
    '-si-peephole-sdwa',
    '-si-shrink-instructions',
    '-si-wqm',
    '-simple-loop-unswitch',
    '-simple-register-coalescing',
    '-simplifycfg'
    '-simplifycfg',
    '-sink',
    '-sjljehprepare',
    '-slotindexes',
    '-slp-vectorizer',
    '-slsr',
    '-speculative-execution',
    '-spill-code-placement',
    '-sroa',
    '-stack-coloring',
    '-stack-protector',
    '-stack-slot-coloring',
    '-stackmap-liveness',
    '-strip',
    '-strip-dead-debug-info',
    '-strip-dead-prototypes',
    '-strip-debug-declare',
    '-strip-gc-relocates',
    '-strip-nondebug',
    '-strip-nonlinetable-debuginfo',
    '-structurizecfg',
    '-t2-reduce-size',
    '-tailcallelim',
    '-tailduplication',
    '-targetlibinfo',
    '-targetpassconfig',
    '-tbaa',
    '-tsan',
    '-tti',
    '-twoaddressinstruction',
    '-unreachable-mbb-elimination',
    '-unreachableblockelim',
    '-vec-merger',
    '-verify',
    '-verify-safepoint-ir',
    '-view-callgraph',
    '-view-cfg',
    '-view-cfg-only',
    '-view-dom',
    '-view-dom-only',
    '-view-postdom',
    '-view-postdom-only',
    '-view-regions',
    '-view-regions-only',
    '-view-scops',
    '-view-scops-only',
    '-virtregmap',
    '-virtregrewriter',
    '-wholeprogramdevirt',
    '-winehprepare',
    '-write-bitcode',
    '-x86-cf-opt',
    '-x86-cmov-conversion',
    '-x86-domain-reassignment',
    '-x86-evex-to-vex-compress',
    '-x86-execution-deps-fix',
    '-x86-fixup-bw-insts',
    '-x86-fixup-LEAs',
    '-x86-winehstate',
    '-xray-instrumentation',
}

# Valid optimization levels. Same as for clang, but without -Ofast.
OPTIMIZATION_LEVELS = {"-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"}


class OptException(llvm.LlvmError):
  """An error from opt."""
  pass


def ValidateOptimizationLevel(opt: str) -> str:
  """Check that the requested optimization level is valid.

  Args:
    opt: The optimization level.

  Returns:
    The input argument.

  Raises:
    ValueError: If optimization level is not valid.
  """
  if opt in OPTIMIZATION_LEVELS:
    return opt
  raise ValueError(f"Invalid opt optimization level '{opt}'. "
                   f"Valid levels are: {OPTIMIZATION_LEVELS}")


def Exec(args: typing.List[str],
         stdin: typing.Optional[typing.Union[str, bytes]] = None,
         timeout_seconds: int = 60,
         universal_newlines: bool = True,
         log: bool = True) -> subprocess.Popen:
  """Run LLVM's optimizer.

  Args:
    args: A list of arguments to pass to binary.
    stdin: Optional input to pass to binary. If universal_newlines is set, this
      should be a string. If not, it should be bytes.
    timeout_seconds: The number of seconds to allow opt to run for.
    universal_newlines: Argument passed to Popen() of opt process.
    log: If true, print executed command to DEBUG log.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(OPT)] + args
  app.LogIf(log, 3, '$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      stdin=subprocess.PIPE if stdin else None,
      universal_newlines=universal_newlines)
  if stdin:
    stdout, stderr = process.communicate(stdin)
  else:
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise llvm.LlvmTimeout(f'clang timed out after {timeout_seconds}s')
  process.stdout = stdout
  process.stderr = stderr
  return process


def RunOptPassOnBytecode(input_path: pathlib.Path,
                         output_path: pathlib.Path,
                         opts: typing.List[str],
                         timeout_seconds: int = 60) -> pathlib.Path:
  """Run opt pass(es) on a bytecode file.

  Args:
    input_path: The input bytecode file.
    output_path: The file to generate.
    opts: Additional flags to pass to opt.
    timeout_seconds: The number of seconds to allow opt to run for.

  Returns:
    The output_path.

  Raises:
    OptException: In case of error.
    LlvmTimeout: If the process times out.
  """
  # We don't care about the output of opt, but we will try and decode it if
  # opt fails.
  proc = Exec(
      [str(input_path), '-o', str(output_path), '-S'] + opts,
      timeout_seconds=timeout_seconds,
      universal_newlines=False)
  if proc.returncode == 9:
    raise llvm.LlvmTimeout(f'opt timed out after {timeout_seconds} seconds')
  elif proc.returncode:
    try:
      stderr = proc.stderr.decode('utf-8')
      raise OptException(
          f'clang exited with returncode {proc.returncode}: {stderr}')
    except UnicodeDecodeError:
      raise OptException(f'clang exited with returncode {proc.returncode}')
  if not output_path.is_file():
    raise OptException(f'Bytecode file {output_path} not generated')
  return output_path


def GetAllOptimizationsAvailable() -> typing.List[str]:
  """Return the full list of optimizations available.

  Returns:
    A list of strings, where each string is an LLVM opt flag to enable an
    optimization.

  Raises:
    OptException: If unable to interpret opt output.
  """
  # We must disable logging here - this function is invoked to set
  # OPTIMIZATION_PASSES variable below, before flags are parsed.
  proc = Exec(['-help-list-hidden'], log=False)
  lines = proc.stdout.split('\n')
  # Find the start of the list of optimizations.
  for i in range(len(lines)):
    if lines[i] == '  Optimizations available:':
      break
  else:
    raise OptException
  # Find the end of the list of optimizations.
  for j in range(i + 1, len(lines)):
    if not lines[j].startswith('    -'):
      break
  else:
    raise OptException

  # Extract the list of optimizations.
  optimizations = [line[len('    '):].split()[0] for line in lines[i + 1:j]]
  if len(optimizations) < 2:
    raise OptException

  return optimizations


@functools.lru_cache(maxsize=1)
def GetAllOptPasses() -> typing.Set[str]:
  """Return all opt passes."""
  return TRANSFORM_PASSES.union(OPTIMIZATION_LEVELS).union(
      set(GetAllOptimizationsAvailable()))


def main(argv):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.opt_timeout_seconds)
    if proc.stdout:
      print(proc.stdout)
    if proc.stderr:
      print(proc.stderr, file=sys.stderr)
    sys.exit(proc.returncode)
  except llvm.LlvmTimeout as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  app.RunWithArgs(main)
