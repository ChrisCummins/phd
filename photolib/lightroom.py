"""Functions for working with Lightroom."""
import typing

from absl import logging
from libxmp import utils as xmputils

# A process-local cache mapping filenames to sets of Lightroom Keywords.
# Access to this cache is not thread safe.
_KEYWORDS_CACHE: typing.Dict[str, typing.Set] = dict()


def get_lightroom_keywords(abspath: str) -> typing.Set[str]:
  """Fetch lightroom."""
  if abspath in _KEYWORDS_CACHE:
    return _KEYWORDS_CACHE[abspath]

  try:
    xmp = xmputils.file_to_dict(abspath)
    lrtags = xmp['http://ns.adobe.com/lightroom/1.0/']
    keywords = set([e[1] for e in lrtags if e[1]])
    _KEYWORDS_CACHE[abspath] = keywords
    return keywords
  except KeyError:
    logging.error(abspath)
    raise KeyError(abspath)
