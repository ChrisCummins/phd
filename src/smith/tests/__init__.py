from unittest import TestCase

import smith

def print_val(obj, **kwargs):
    if hasattr(obj, "__iter__"):
        if isinstance(obj, dict):
            print(obj, **kwargs)
        elif not isinstance(obj, str):
            print([str(x) for x in obj], **kwargs)
    else:
        print(str(obj), **kwargs)


class TestSmith(TestCase):
    def test_foo(self):
        self.assertEqual(5, smith.foo())


if __name__ == '__main__':
    main()
