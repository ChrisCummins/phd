from unittest import TestCase

import smith

class TestSmith(TestCase):
    def test_get_substring_idxs(self):
        self.assertEqual([0, 2], smith.get_substring_idxs('a', 'aba'))
        self.assertEqual([], smith.get_substring_idxs('a', 'bb'))

if __name__ == '__main__':
    main()
