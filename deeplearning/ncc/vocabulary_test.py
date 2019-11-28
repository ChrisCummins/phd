"""Unit tests for //deeplearning/ncc:vocabulary."""
import pytest

from deeplearning.ncc import inst2vec_pb2
from deeplearning.ncc import vocabulary
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

FLAGS = app.FLAGS

VOCABULARY_PATH = bazelutil.DataPath(
  "phd/deeplearning/ncc/published_results/vocabulary.zip"
)

# LLVM IR for the following C function:
#
#   int FizzBuzz(int i) {
#     if (i % 15 == 0) {
#       return 1;
#     }
#     return 0;
#   }
#
# Generated using:
#
#   bazel run //compilers/llvm:clang -- /tmp/fizzbuzz.c -- \
#       -emit-llvm -S -o /tmp/fizzbuzz.ll
#
FIZZBUZZ_IR = """\
; ModuleID = '/tmp/fizzbuzz.c'
source_filename = "/tmp/fizzbuzz.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @FizzBuzz(i32) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  %5 = srem i32 %4, 15
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %8

; <label>:7:                                      ; preds = %1
  store i32 1, i32* %2, align 4
  br label %9

; <label>:8:                                      ; preds = %1
  store i32 0, i32* %2, align 4
  br label %9

; <label>:9:                                      ; preds = %8, %7
  %10 = load i32, i32* %2, align 4
  ret i32 %10
}

attributes #0 = { noinline nounwind optnone ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
"""


@test.Fixture(scope="session")
def vocab() -> vocabulary.VocabularyZipFile:
  """Test fixture which yields a vocabulary zip file instance as a ctx mngr."""
  with vocabulary.VocabularyZipFile(VOCABULARY_PATH) as v:
    yield v


def test_VocabularyZipFile_dictionary_type(vocab: vocabulary.VocabularyZipFile):
  """Test that dictionary is a dict."""
  assert isinstance(vocab.dictionary, dict)


def test_VocabularyZipFile_dictionary_size(vocab: vocabulary.VocabularyZipFile):
  """Test that dictionary contains at least 2 values (1 for !UNK, +1 other)."""
  assert len(vocab.dictionary) >= 2


def test_VocabularyZipFile_dictionary_values_are_unique(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that values in vocabulary are unique."""
  assert len(set(vocab.dictionary.values())) == len(vocab.dictionary.values())


def test_VocabularyZipFile_dictionary_values_are_positive_integers(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that values in vocabulary are unique."""
  for value in vocab.dictionary.values():
    assert value >= 0


def test_VocabularyZipFile_cutoff_stmts_type(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that cutoff_stmts is a set."""
  assert isinstance(vocab.cutoff_stmts, set)


def test_VocabularyZipFile_unknown_token_index_type(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that unknown token index is an integer."""
  assert isinstance(vocab.unknown_token_index, int)
  assert vocab.unknown_token_index > 0


def test_VocabularyZipFile_unknown_token_index_value(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that unknown token index is positive."""
  assert vocab.unknown_token_index > 0


def test_VocabularyZipFile_EncodeLlvmBytecode_bytecode(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that bytecode is set in return value."""
  result = vocab.EncodeLlvmBytecode(FIZZBUZZ_IR)
  assert result.input_bytecode == FIZZBUZZ_IR


def test_VocabularyZipFile_EncodeLlvmBytecode_preprocessing(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test output of pre-processing bytecode."""
  options = inst2vec_pb2.EncodeBytecodeOptions(
    set_bytecode_after_preprocessing=True
  )
  result = vocab.EncodeLlvmBytecode(FIZZBUZZ_IR, options)

  assert (
    result.bytecode_after_preprocessing
    == """\
define i32 <@ID>(i32)
<%ID> = alloca i32, align 4
<%ID> = alloca i32, align 4
store i32 <%ID>, i32* <%ID>, align 4
<%ID> = load i32, i32* <%ID>, align 4
<%ID> = srem i32 <%ID>, <INT>
<%ID> = icmp eq i32 <%ID>, <INT>
br i1 <%ID>, label <%ID>, label <%ID>
; <label>:<LABEL>: ; preds = <LABEL>
store i32 <INT>, i32* <%ID>, align 4
br label <%ID>
; <label>:<LABEL>: ; preds = <LABEL>
store i32 <INT>, i32* <%ID>, align 4
br label <%ID>
; <label>:<LABEL>: ; preds = <LABEL>, <LABEL>
<%ID> = load i32, i32* <%ID>, align 4
ret i32 <%ID>"""
  )


# LLVM IR for the following C code:
#
#     struct Foo {
#       int x;
#       int y;
#     };
#
#     void DoFoo(struct Foo* foo) {
#       foo->x += foo->y;
#     }
#
# Generated using:
#
#   bazel run //compilers/llvm:clang -- /tmp/foo.c -- \
#       -emit-llvm -S -o /tmp/foo.ll
#
BYTECODE_WITH_STRUCT = """\
; ModuleID = 'foo.c'
source_filename = "foo.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.Foo = type { i32, i32 }

; Function Attrs: nounwind ssp uwtable
define void @DoFoo(%struct.Foo*) #0 {
  %2 = alloca %struct.Foo*, align 8
  store %struct.Foo* %0, %struct.Foo** %2, align 8
  %3 = load %struct.Foo*, %struct.Foo** %2, align 8
  %4 = getelementptr inbounds %struct.Foo, %struct.Foo* %3, i32 0, i32 1
  %5 = load i32, i32* %4, align 4
  %6 = load %struct.Foo*, %struct.Foo** %2, align 8
  %7 = getelementptr inbounds %struct.Foo, %struct.Foo* %6, i32 0, i32 0
  %8 = load i32, i32* %7, align 4
  %9 = add nsw i32 %8, %5
  store i32 %9, i32* %7, align 4
  ret void
}

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"Apple LLVM version 8.0.0 (clang-800.0.42.1)"}
"""


def test_VocabularyZipFile_EncodeLlvmBytecode_struct_dict(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that struct appears in struct_dict."""
  options = inst2vec_pb2.EncodeBytecodeOptions(set_struct_dict=True)
  result = vocab.EncodeLlvmBytecode(BYTECODE_WITH_STRUCT, options)

  assert dict(result.struct_dict) == {"%struct.Foo": "{ i32, i32 }"}


def test_VocabularyZipFile_EncodeLlvmBytecode_struct_not_in_preprocessed(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test that struct is inlined during pre-processing."""
  options = inst2vec_pb2.EncodeBytecodeOptions(
    set_bytecode_after_preprocessing=True
  )
  result = vocab.EncodeLlvmBytecode(BYTECODE_WITH_STRUCT, options)

  assert (
    result.bytecode_after_preprocessing
    == """\
define void <@ID>({ i32, i32 }*)
<%ID> = alloca { i32, i32 }*, align 8
store { i32, i32 }* <%ID>, { i32, i32 }** <%ID>, align 8
<%ID> = load { i32, i32 }*, { i32, i32 }** <%ID>, align 8
<%ID> = getelementptr inbounds { i32, i32 }, { i32, i32 }* <%ID>, i32 <INT>, i32 <INT>
<%ID> = load i32, i32* <%ID>, align 4
<%ID> = load { i32, i32 }*, { i32, i32 }** <%ID>, align 8
<%ID> = getelementptr inbounds { i32, i32 }, { i32, i32 }* <%ID>, i32 <INT>, i32 <INT>
<%ID> = load i32, i32* <%ID>, align 4
<%ID> = add nsw i32 <%ID>, <%ID>
store i32 <%ID>, i32* <%ID>, align 4
ret void"""
  )


def test_VocabularyZipFile_EncodeLlvmBytecode_encode_single_line(
  vocab: vocabulary.VocabularyZipFile,
):
  """Test encoding a single line of bytecode."""
  # A single line of bytecode which references a struct.
  result = vocab.EncodeLlvmBytecode(
    "store %struct.Foo* %0, %struct.Foo** %2, align 8",
    options=inst2vec_pb2.EncodeBytecodeOptions(
      set_bytecode_after_preprocessing=True,
    ),
    struct_dict={"%struct.Foo": "{ i32, i32 }"},
  )

  assert (
    result.bytecode_after_preprocessing
    == "store { i32, i32 }* <%ID>, { i32, i32 }** <%ID>, align 8"
  )


def test_VocabularyZipFile_EncodeLlvmBytecode_sequence(
  vocab: vocabulary.VocabularyZipFile,
):
  # Function contains 14 statements.
  result = vocab.EncodeLlvmBytecode(FIZZBUZZ_IR)
  assert len(result.encoded) == 14


if __name__ == "__main__":
  test.Main()
