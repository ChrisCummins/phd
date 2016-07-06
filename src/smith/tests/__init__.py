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
    def test_pacakge_data_404(self):
        with self.assertRaises(smith.InternalException):
            smith.package_data("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.package_data("This definitely isn't a real path")

    def test_pacakge_str_404(self):
        with self.assertRaises(smith.InternalException):
            smith.package_str("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.package_str("This definitely isn't a real path")

    def test_sql_script_404(self):
        with self.assertRaises(smith.InternalException):
            smith.sql_script("This definitely isn't a real path")
        with self.assertRaises(smith.Data404Exception):
            smith.sql_script("This definitely isn't a real path")


if __name__ == '__main__':
    main()
