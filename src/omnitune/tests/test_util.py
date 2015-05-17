from unittest import main
from tests import TestCase

import omnitune
from omnitune import util

class TestUtil(TestCase):

    # parse_str()
    def test_parse_str(self):
        self._test("abd", util.parse_str("abc"))


if __name__ == '__main__':
    main()
