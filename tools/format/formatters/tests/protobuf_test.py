# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //tools/format/formatters:shell."""
from labm8.py import test
from tools.format.formatters import protobuf

FLAGS = test.FLAGS


def test_invalid_empty_file():
  with test.Raises(protobuf.FormatProtobuf.FormatError):
    protobuf.FormatProtobuf.Format("")


def test_invalid_proto_file_missing_syntax():
  with test.Raises(protobuf.FormatProtobuf.FormatError):
    protobuf.FormatProtobuf.Format(
      """
message Foo { optional int32 a = 1; }
"""
    )


def test_invalid_proto_field_missing_tag_number():
  with test.Raises(protobuf.FormatProtobuf.FormatError):
    protobuf.FormatProtobuf.Format(
      """
syntax = "proto2";
message Foo { optional int32 a; }
"""
    )


def test_proto_with_missing_import():
  with test.Raises(protobuf.FormatProtobuf.FormatError):
    protobuf.FormatProtobuf.Format(
      """
syntax = "proto2";
import "this/is/not/found.proto";
message Foo { optional int32 a; }"""
    )


def test_small_proto2_file():
  text = protobuf.FormatProtobuf.Format(
    """
syntax = "proto2";
message Foo { optional int32 a = 1; }"""
  )
  print(text)
  assert (
    text
    == """\
syntax = "proto2";

message Foo {
  optional int32 a = 1;
}
"""
  )


def test_small_proto3_file():
  text = protobuf.FormatProtobuf.Format(
    """
  syntax = "proto3";
  message Foo { int32 a = 1; }"""
  )
  print(text)
  assert (
    text
    == """\
syntax = "proto3";

message Foo {
  int32 a = 1;
}
"""
  )


if __name__ == "__main__":
  test.Main()
