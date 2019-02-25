def hash(key: str) -> int:
  return int(key) % 10


class HashMap:

  def __init__(self, nbuckets=10):
    self.buckets = dict((k, []) for k in range(nbuckets))

  def insert(self, key: str, value):
    hash_ = hash(key)
    if hash_ in self.buckets[hash_]:
      raise KeyError
    self.buckets[hash_].append((key, value))
