"""Unit tests for //deeplearning/ncc/inst2vec:inst2vec_preprocess."""
import pathlib
import tempfile

import pytest

from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope="function")
def data_folder() -> str:
  """A test fixture that produces a data folder with a single bytecode."""

  with tempfile.TemporaryDirectory(prefix="phd_") as d:
    data_folder = pathlib.Path(d)
    (data_folder / "BLAS-3.8.0" / "blas").mkdir(parents=True)

    # A single file from the BLAS dataset, downloaded from
    # https://polybox.ethz.ch/index.php/s/5ASMNv6dYsPKjyQ/download
    with open(data_folder / "BLAS-3.8.0" / "blas" / "fast_dcabs1.ll", "w") as f:
      f.write(
        """\
; ModuleID = '/tmp/dcabs1-c19f92.ll'
source_filename = "/tmp/dcabs1-c19f92.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
define double @dcabs1_(i64* nocapture readonly %z) local_unnamed_addr #0 !dbg !5 {
L.entry:
  %0 = bitcast i64* %z to <2 x double>*, !dbg !11
  %1 = load <2 x double>, <2 x double>* %0, align 1, !dbg !11, !tbaa !13
  %2 = call <2 x double> @llvm.fabs.v2f64(<2 x double> %1), !dbg !11
  %3 = extractelement <2 x double> %2, i32 0, !dbg !11
  %4 = extractelement <2 x double> %2, i32 1, !dbg !11
  %5 = fadd fast double %4, %3, !dbg !11
  ret double %5, !dbg !17
}

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #1

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "dcabs1.f", directory: "/home/shoshijak/Documents/blas_ir/BLAS-3.8.0")
!4 = !{}
!5 = distinct !DISubprogram(name: "dcabs1", scope: !2, file: !3, line: 48, type: !6, isLocal: false, isDefinition: true, scopeLine: 48, isOptimized: false, unit: !2, variables: !4)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!9 = !DIBasicType(name: "double precision", size: 64, align: 64, encoding: DW_ATE_float)
!10 = !DIBasicType(name: "double complex", size: 128, align: 64, encoding: DW_ATE_complex_float)
!11 = !DILocation(line: 64, column: 1, scope: !12)
!12 = !DILexicalBlock(scope: !5, file: !3, line: 48, column: 1)
!13 = !{!14, !14, i64 0}
!14 = !{!"t1.2", !15, i64 0}
!15 = !{!"unlimited ptr", !16, i64 0}
!16 = !{!"Flang FAA 1"}
!17 = !DILocation(line: 66, column: 1, scope: !12)\
"""
      )

    # Yield the temporary directory path as a string.
    yield d


def test_CreateContextualFlowGraphsFromBytecodes_files(data_folder: str):
  """Test that expected files are produced."""
  inst2vec_preprocess.CreateContextualFlowGraphsFromBytecodes(data_folder)
  d = pathlib.Path(data_folder)

  # Files produced:
  assert (d / "BLAS-3.8.0/blas/data_read_pickle").is_file()
  assert (d / "BLAS-3.8.0/blas_preprocessed/data_preprocessed_pickle").is_file()

  # Folders produced:
  assert (d / "BLAS-3.8.0/blas_preprocessed/data_transformed").is_dir()
  assert (d / "BLAS-3.8.0/blas_preprocessed/preprocessed").is_dir()
  assert (d / "BLAS-3.8.0/blas_preprocessed/structure_dictionaries").is_dir()
  assert (d / "BLAS-3.8.0/blas_preprocessed/xfg").is_dir()
  assert (d / "BLAS-3.8.0/blas_preprocessed/xfg_dual").is_dir()

  # Files in folders:
  assert (
    d / "BLAS-3.8.0/blas_preprocessed/data_transformed/fast_dcabs1.p"
  ).is_file()
  assert (
    d / "BLAS-3.8.0/blas_preprocessed/preprocessed/fast_dcabs1_preprocessed.txt"
  ).is_file()
  assert (
    d / "BLAS-3.8.0/blas_preprocessed/structure_dictionaries/fast_dcabs1.txt"
  ).is_file()
  assert (d / "BLAS-3.8.0/blas_preprocessed/xfg/fast_dcabs1.txt").is_file()
  assert (d / "BLAS-3.8.0/blas_preprocessed/xfg_dual/fast_dcabs1.p").is_file()


if __name__ == "__main__":
  test.Main()
