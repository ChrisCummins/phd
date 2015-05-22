from unittest import main
from tests import TestCase

import omnitune
from omnitune import cache

class TestCache(TestCase):

    def _test_cache(self, c):
        c.clear()
        self._test(False, c.contains("foo"))
        self._test(1, c.set("foo", 1))
        self._test(True, c.contains("foo"))
        self._test(1, c.get("foo"))
        c.remove("foo")
        self._test(False, c.contains("foo"))
        c.clear()
        self._test(1, c.set("foo", 1))
        self._test(1, c.get("foo"))
        c.clear()

    # TransientCache
    def test_transient_cache(self):
        c = cache.TransientCache()
        self._test_cache(c)

    # JsonCache
    def test_json_cache(self):
        c = cache.JsonCache("tests/data/jsoncache.json")

        self._test(True, c.contains("foo"))
        self._test(1, c.get("foo"))
        self._test_cache(c)
        self._test(1, c.set("foo", 1))


if __name__ == '__main__':
    main()
