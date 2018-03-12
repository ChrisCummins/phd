import google.protobuf.json_format
import google.protobuf.text_format
import gzip
import pathlib
import typing

# A type alias for annotating methods which take or return protocol buffers.
ProtocolBuffer = typing.Any


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
  suffixes = path.suffixes
  if suffixes and suffixes[-1] == '.gz':
    suffixes.pop()
    open_function = gzip.open
  else:
    open_function = open

  suffix = suffixes[-1] if suffixes else ''
  try:
    with open_function(path, 'rb') as f:
      if suffix == '.txt':
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
  mode = 'wt' if suffix in {'.txt', '.json'} else 'wb'

  with open_function(path, mode) as f:
    if suffix == '.txt':
      f.write(google.protobuf.text_format.MessageToString(message))
    elif suffix == '.json':
      f.write(google.protobuf.json_format.MessageToJson(message))
    else:
      f.write(message.SerializeToString())

  return message
