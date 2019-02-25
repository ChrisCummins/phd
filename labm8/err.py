"""Errors and assertions helpers.
"""


def assert_or_raise(stmt: bool, exception: Exception, *exception_args,
                    **exception_kwargs) -> None:
  """
  If the statement is false, raise the given exception.
  """
  if not stmt:
    raise exception(*exception_args, **exception_kwargs)
