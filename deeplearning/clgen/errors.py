class CLgenError(Exception):
  """
  Top level error. Never directly thrown.
  """
  pass


class InternalError(CLgenError):
  """
  An internal module error. This class of errors should not leak outside of
  the module into user code.
  """
  pass


class UserError(CLgenError):
  """
  Raised in case of bad user interaction, e.g. an invalid argument.
  """
  pass


class File404(InternalError):
  """
  Data not found.
  """
  pass


class InvalidFile(UserError):
  """
  Raised in case a file contains invalid contents.
  """
  pass
