import atexit
import json

import labm8
from labm8 import fs
from labm8 import io

class Cache(object):

    def contains(self, key):
        pass

    def get(self, key, default=None):
        pass

    def set(self, key, value):
        pass

    def clear(self):
        pass


class TransientCache(Cache):

    def __init__(self):
        self._cache = {}

    def contains(self, key):
        return key in self._cache

    def get(self, key, default=None):
        if key in self._cache:
            return self._cache[key]
        else:
            return default

    def set(self, key, value):
        self._cache[key] = value
        return value

    def remove(self, key):
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        self._cache.clear()


class JsonCache(TransientCache):

    def write(self):
        """
        Write cache to disk.
        """
        io.debug("Storing cache '{0}'".format(self.path))
        with open(self.path, "w") as file:
            json.dump(self._cache, file, sort_keys=True, indent=2,
                      separators=(',', ': '))


    def __init__(self, path):
        super(JsonCache, self).__init__()
        self.path = fs.path(path)

        if fs.exists(self.path):
            io.debug(("Loading cache '{0}'".format(self.path)))
            # Try to load cache from disk.
            with open(self.path) as file:
                self._cache = json.load(file)

        # Register exit handler
        atexit.register(self.write)
