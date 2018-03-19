import collections

import google.protobuf.json_format
import google.protobuf.text_format
import gzip
import json
import pathlib
import typing

# A type alias for annotating methods which take or return protocol buffers.
ProtocolBuffer = typing.Any

# A type alias for JSON data.
JSON = typing.Dict[str, typing.Any]


class EncodeError(Exception):
  pass


class DecodeError(Exception):
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


def ToJson(message: ProtocolBuffer) -> JSON:
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


def _TruncateDictionaryStringValues(data: JSON, n: int = 62) -> JSON:
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


def RaiseIfNotSet(proto: ProtocolBuffer, field: str) -> typing.Any:
  """Check that a proto field is set before returning it.

  Args:
    proto: A message instance.
    field: The name of the field.

  Returns:
    The value of the field.

  Raises:
    ValueError: If the field is not set.
  """
  if not proto.HasField(field):
    raise ValueError(f'datastore field {field} not set')
  elif not getattr(proto, field):
    raise ValueError(f'datastore field {field} not set')
  return getattr(proto, field)
