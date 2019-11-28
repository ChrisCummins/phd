"""Unit tests for //deeplearning/ncc:task_utils."""
import pathlib

import pytest

from deeplearning.ncc import task_utils
from deeplearning.ncc import vocabulary
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

FLAGS = app.FLAGS

VOCABULARY_PATH = bazelutil.DataPath(
  "phd/deeplearning/ncc/published_results/vocabulary.zip"
)

# An example LLVM IR, taken from the SHOC benchmark suite.
EXAMPLE_OPENCL_IR = """\
; ModuleID = 'shoc-1.1.5-Triad-Triad.cl'
source_filename = "shoc-1.1.5-Triad-Triad.cl"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

; Function Attrs: nounwind ssp uwtable
define spir_kernel void @Triad(float* nocapture readonly, float* nocapture readonly, float* nocapture, float) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  %5 = tail call i64 @_Z13get_global_idj(i32 0) #3
  %6 = shl i64 %5, 32
  %7 = ashr exact i64 %6, 32
  %8 = getelementptr inbounds float, float* %0, i64 %7
  %9 = load float, float* %8, align 4, !tbaa !8
  %10 = getelementptr inbounds float, float* %1, i64 %7
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %11, float %9)
  %13 = getelementptr inbounds float, float* %2, i64 %7
  store float %12, float* %13, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @_Z13get_global_idj(i32) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind readnone }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 1, i32 0}
!3 = !{!"Apple LLVM version 9.1.0 (clang-902.0.39.1)"}
!4 = !{i32 1, i32 1, i32 1, i32 0}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"float*", !"float"}
!7 = !{!"const", !"const", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
"""


@test.Fixture(scope="function")
def llvm_ir_dir(tempdir: pathlib.Path) -> str:
  """A test fixture which returns the path to a directory containing LLVM IR."""
  (tempdir / "llvm_ir").mkdir()
  with open(tempdir / "llvm_ir" / "program.ll", "w") as f:
    f.write(EXAMPLE_OPENCL_IR)
  yield str(tempdir / "llvm_ir")


@test.Fixture(scope="session")
def vocab() -> vocabulary.VocabularyZipFile:
  """Test fixture which yields a vocabulary zip file instance as a ctx mngr."""
  with vocabulary.VocabularyZipFile(VOCABULARY_PATH) as v:
    yield v


def test_CreateSeqDirFromIr_creates_directory(
  llvm_ir_dir: str, vocab: vocabulary.VocabularyZipFile
):
  """Test that sequence directory is returned."""
  sequence_folder = pathlib.Path(
    task_utils.CreateSeqDirFromIr(llvm_ir_dir, vocab)
  )
  assert sequence_folder.is_dir()


def test_CreateSeqDirFromIr_creates_csv_file(
  llvm_ir_dir: str, vocab: vocabulary.VocabularyZipFile
):
  """Test that CSV file is created."""
  sequence_folder = pathlib.Path(
    task_utils.CreateSeqDirFromIr(llvm_ir_dir, vocab)
  )
  assert (sequence_folder / "program_seq.csv").is_file()


def test_CreateSeqDirFromIr_creates_rec_file(
  llvm_ir_dir: str, vocab: vocabulary.VocabularyZipFile
):
  """Test that REC file is created."""
  sequence_folder = pathlib.Path(
    task_utils.CreateSeqDirFromIr(llvm_ir_dir, vocab)
  )
  assert (sequence_folder / "program_seq.rec").is_file()


if __name__ == "__main__":
  test.Main()
