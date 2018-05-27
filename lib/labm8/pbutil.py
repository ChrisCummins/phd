"""Utility code for working with Protocol Buffers."""
import collections
import gzip
import json
import pathlib
import typing

import google.protobuf.json_format
import google.protobuf.text_format

from lib.labm8 import jsonutil


# A type alias for annotating methods which take or return protocol buffers.
ProtocolBuffer = typing.Any


class ProtoValueError(ValueError):
  """Raised in case of a value error from a proto."""
  pass


class EncodeError(ProtoValueError):
  """Raised in case of error encoding a proto."""
  pass


class DecodeError(ProtoValueError):
  """Raised in case of error decoding a proto."""
  pass


def FromFile(path: pathlib.Path, message: ProtocolBuffer) -> ProtocolBuffer:
  """Read a protocol buffer from a file.

  This method uses attempts to guess the encoding from the path suffix,
  supporting binary, text, and json formatted messages. The mapping of suffixes
  to formatting is, in order:
      *.txt.gz: Gzipped text.
      *.txt: Text.
      *.pbtxt.gz: Gzipped text.
      *.pbtxt: Text.
      *.json.gz: Gzipped JSON.
      *.json: JSON.
      *.gz: Gzipped encoded string.
      *: Encoded string.

  Args:
    path: Path to the proto file.
    message: A message instance to read into.

  Returns:
    The parsed message (same as the message argument).

  Raises:
    IOError: If the file does not exist or cannot be read.
    DecodeError: If the file cannot be decoded to the given message type. Note
      that parsing from binary encoding (i.e. not *.txt or *.json) does not
      raise this error. Instead, unknown fields are silently ignored.
  """
  if not path.is_file():
    raise IOError(f'Not a file: {path}')

  suffixes = path.suffixes
  if suffixes and suffixes[-1] == '.gz':
    suffixes.pop()
    open_function = gzip.open
  else:
    open_function = open

  suffix = suffixes[-1] if suffixes else ''
  try:
    with open_function(path, 'rb') as f:
      if suffix == '.txt' or suffix == '.pbtxt':
        google.protobuf.text_format.Merge(f.read(), message)
      elif suffix == '.json':
        google.protobuf.json_format.Parse(f.read(), message)
      else:
        message.ParseFromString(f.read())
  except (google.protobuf.text_format.ParseError,
          google.protobuf.json_format.ParseError) as e:
    # The exception raised during parsing depends on the message format. Catch
    # them all under a single DecodeError exception type.
    raise DecodeError(e)

  return message


def ToFile(message: ProtocolBuffer, path: pathlib.Path,
           exist_ok: bool = True) -> ProtocolBuffer:
  """Write a protocol buffer to a file.

  This method uses attempts to guess the encoding from the path suffix,
  supporting binary, text, and json formatted messages. The mapping of suffixes
  to formatting is, in order:
      *.txt.gz: Gzipped text.
      *.txt: Text.
      *.pbtxt.gz: Gzipped text.
      *.pbtxt: Text.
      *.json.gz: Gzipped JSON.
      *.json: JSON.
      *.gz: Gzipped encoded string.
      *: Encoded string.

  Args:
    message: A message instance to write to file.
    path: Path to the proto file.
    exist_ok: If True, overwrite existing file.

  Returns:
    The parsed message (same as the message argument).

  Raises:
    IOError: If exist_ok is False and file already exists.
  """
  if not exist_ok and path.exists():
    raise IOError(f'Refusing to overwrite {path}')

  suffixes = path.suffixes
  if suffixes and suffixes[-1] == '.gz':
    suffixes.pop()
    open_function = gzip.open
  else:
    open_function = open

  suffix = suffixes[-1] if suffixes else ''
  mode = 'wt' if suffix in {'.txt', '.pbtxt', '.json'} else 'wb'

  with open_function(path, mode) as f:
    if suffix == '.txt' or suffix == '.pbtxt':
      f.write(google.protobuf.text_format.MessageToString(message))
    elif suffix == '.json':
      f.write(google.protobuf.json_format.MessageToJson(
          message, preserving_proto_field_name=True))
    else:
      f.write(message.SerializeToString())

  return message


def ToJson(message: ProtocolBuffer) -> jsonutil.JSON:
  """Return a JSON encoded representation of a protocol buffer.

  Args:
    message: The message to convert to JSON.

  Returns:
    JSON encoded message.
  """
  return google.protobuf.json_format.MessageToDict(
      message, preserving_proto_field_name=True)


def _TruncatedString(string: str, n: int = 80) -> str:
  """Return the truncated first 'n' characters of a string.

  Args:
    string: The string to truncate.
    n: The maximum length of the string to return.

  Returns:
    The truncated string.
  """
  if len(string) > n:
    return string[:n - 3] + '...'
  else:
    return string


def _TruncateDictionaryStringValues(
    data: jsonutil.JSON, n: int = 62) -> jsonutil.JSON:
  """Truncate all string values in a nested dictionary.

  Args:
    data: A dictionary.

  Returns:
    The dictionary.
  """
  for key, value in data.items():
    if isinstance(value, collections.Mapping):
      data[key] = _TruncateDictionaryStringValues(data[key])
    elif isinstance(value, str):
      data[key] = _TruncatedString(value, n)
    else:
      data[key] = value
  return data


def PrettyPrintJson(message: ProtocolBuffer,
                    truncate: int = 52) -> str:
  """Return a pretty printed JSON string representation of the message.

  Args:
    message: The message to pretty print.
    truncate: The length to truncate string values. Truncation is disabled if
      this argument is None.

  Returns:
    JSON string.
  """
  data = ToJson(message)
  return json.dumps(_TruncateDictionaryStringValues(data) if truncate else data,
                    indent=2, sort_keys=True)


def RaiseIfNotSet(proto: ProtocolBuffer, field: str,
                  err: ValueError) -> typing.Any:
  """Check that a proto field is set before returning it.

  Args:
    proto: A message instance.
    field: The name of the field.
    err: The exception class to raise.

  Returns:
    The value of the field.

  Raises:
    ValueError: If the field is not set.
  """
  if not proto.HasField(field):
    raise err(f'datastore field {field} not set')
  elif not getattr(proto, field):
    raise err(f'datastore field {field} not set')
  return getattr(proto, field)


def ProtoIsReadable(path: typing.Union[str, pathlib.Path],
                    message: ProtocolBuffer) -> bool:
  """Return whether a file is a readable protocol buffer.

  Arguments:
    path: The path of the file to read.
    message: An instance of the message type.

  Returns:
    True if contents of path can be parsed as an instance of message, else
    False.
  """
  try:
    FromFile(pathlib.Path(path), message)
    return True
  except:
    return False


def AssertFieldIsSet(proto: ProtocolBuffer, field_name: str,
                     fail_message: str = None) -> typing.Optional[typing.Any]:
  """Assert that protocol buffer field is set.

  Args:
    proto: A proto message instance.
    field_name: The name of the field to assert the constraint on.
    fail_message: An optional message to raise the ProtoValueError
      with if the assertion fails. If not provided, a default message is used.

  Returns:
    The value of the field, if the field has a value. Even though a field may
      be set, it may not have a value. For example, if any of a 'oneof' fields
      is set, then this function will return True for the name of the oneof,
      but the return value will be None.

  Raises:
    ValueError: If the requested field does not exist in the proto schema.
    ProtoValueError: If the field is not set.
  """
  if not proto.HasField(field_name):
    proto_class_name = type(proto).__name__
    raise ProtoValueError(
        fail_message or f"Field not set: '{proto_class_name}.{field_name}'")
  return getattr(proto, field_name) if hasattr(proto, field_name) else None


def AssertFieldConstraint(proto: ProtocolBuffer, field_name: str,
                          constraint: typing.Callable[
                            [typing.Any], bool] = lambda x: True,
                          fail_message: str = None) -> typing.Optional[
  typing.Any]:
  """Assert a constraint on the value of a protocol buffer field.

  Args:
    proto: A proto message instance.
    field_name: The name of the field to assert the constraint on.
    constraint: A constraint checking function to call with the value of the
      field. The function must return True if the constraint check passes, else
      False. If no constraint is specified, this callback always returns True.
      This still allows you to use this function to check if a field is set.
    fail_message: An optional message to raise the ProtoValueError
      with if the assertion fails. If not provided, default messages are used.

  Returns:
    The value of the field.

  Raises:
    ValueError: If the requested field does not exist in the proto schema.
    ProtoValueError: If the field is not set, or if the constraint callback
      returns False for the field's value.
  """
  value = AssertFieldIsSet(proto, field_name, fail_message)
  if not constraint(value):
    proto_class_name = type(proto).__name__
    raise ProtoValueError(
        fail_message or
        f"Field fails constraint check: '{proto_class_name}.{field_name}'")
  else:
    return value
