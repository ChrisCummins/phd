"""A python wrapper around opt, the LLVM optimizer.

opt is part of the LLVM compiler infrastructure. See: http://llvm.org.

This file can be executed as a binary in order to invoke opt. Note you must
use '--' to prevent this script from attempting to parse the args, and a
second '--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/llvm:opt [-- <script_args> [-- <opt_args>]]
"""
import pathlib
import subprocess
import sys
import typing
from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import system

from compilers.llvm import llvm


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'opt_timeout_seconds', 60,
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

OPTIMIZATION_LEVELS = {
  '-O0',
  '-O1',
  '-O2',
  '-O3',
}

ALL_PASSES = TRANSFORM_PASSES.union(OPTIMIZATION_LEVELS)


class OptException(llvm.LlvmError):
  """An error from opt."""
  pass


def Exec(args: typing.List[str], timeout_seconds: int = 60,
         universal_newlines: bool = True) -> subprocess.Popen:
  """Run LLVM's optimizer.

  Args:
    args: A list of arguments to pass to binary.
    timeout_seconds: The number of seconds to allow opt to run for.
    universal_newlines: Argument passed to Popen() of opt process.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(OPT)] + args
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=universal_newlines)
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
  proc = Exec([str(input_path), '-o', str(output_path), '-S'] + opts,
              timeout_seconds=timeout_seconds, universal_newlines=False)
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
  app.run(main)
