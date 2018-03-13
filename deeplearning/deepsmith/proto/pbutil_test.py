"""Unit tests for //deeplearning.deepsmith.proto.pbutil."""
import sys
import tempfile

import google.protobuf.message
import gzip
import json
import pathlib
import pytest
from absl import app

from deeplearning.deepsmith.proto import pbutil
from deeplearning.deepsmith.proto import test_protos_pb2


def test_ToFile_missing_required():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.gz') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(number=1)
    with pytest.raises(google.protobuf.message.EncodeError):
      pbutil.ToFile(proto_in, path)


def test_ToFile_bad_path():
  proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
  with pytest.raises(IOError):
    pbutil.ToFile(proto_in, pathlib.Path('/not a real/path/I think/'))


def test_FromFile_wrong_message_type_txt():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.txt') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    proto_out = test_protos_pb2.AnotherTestMessage()
    with pytest.raises(pbutil.DecodeError):
      pbutil.FromFile(path, proto_out)

  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.pbtxt') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    proto_out = test_protos_pb2.AnotherTestMessage()
    with pytest.raises(pbutil.DecodeError):
      pbutil.FromFile(path, proto_out)


def test_FromFile_wrong_message_type_json():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.json') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    proto_out = test_protos_pb2.AnotherTestMessage()
    with pytest.raises(pbutil.DecodeError):
      pbutil.FromFile(path, proto_out)


def test_FromFile_wrong_message_type_binary():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    proto_out = test_protos_pb2.AnotherTestMessage()
    pbutil.FromFile(path, proto_out)
    # Parsing from binary format does not yield an error on unknown fields.
    assert proto_out.number == 0


def test_ToFile_FromFile_equivalence_txt():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.txt') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with open(path) as f:
      text = f.read()
    assert text == 'string: "abc"\nnumber: 1\n'
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out

  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.pbtxt') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with open(path) as f:
      text = f.read()
    assert text == 'string: "abc"\nnumber: 1\n'
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def test_ToFile_FromFile_equivalence_json():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.json') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with open(path, 'rb') as f:
      json_out = json.load(f)
    assert json_out['string'] == 'abc'
    assert json_out['number'] == 1
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def test_ToFile_FromFile_equivalence_binary():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def test_ToFile_FromFile_equivalence_txt_gz():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.txt.gz') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with gzip.open(path, 'r') as f:
      text = f.read().decode('utf-8')
    assert text == 'string: "abc"\nnumber: 1\n'
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out

  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.pbtxt.gz') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with gzip.open(path, 'r') as f:
      text = f.read().decode('utf-8')
    assert text == 'string: "abc"\nnumber: 1\n'
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def test_ToFile_FromFile_equivalence_json_gz():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.json.gz') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    with gzip.open(path, 'rb') as f:
      json_out = json.load(f)
    assert json_out['string'] == 'abc'
    assert json_out['number'] == 1
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def test_ToFile_FromFile_equivalence_binary_gz():
  with tempfile.NamedTemporaryFile(prefix='deeplearning.deepsmith.proto.',
                                   suffix='.gz') as d:
    path = pathlib.Path(d.name)
    proto_in = test_protos_pb2.TestMessage(string='abc', number=1)
    pbutil.ToFile(proto_in, path)
    assert path.is_file()
    proto_out = test_protos_pb2.TestMessage()
    assert proto_in != proto_out  # Sanity check.
    pbutil.FromFile(path, proto_out)
    assert proto_out.string == 'abc'
    assert proto_out.number == 1
    assert proto_in == proto_out


def main(argv):
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
