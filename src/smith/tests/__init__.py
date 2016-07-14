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
    def test_checksum(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                          smith.checksum("foo".encode()))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                          smith.checksum("bar".encode()))

    def test_checksum_str(self):
        self.assertEqual("0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                          smith.checksum_str("foo"))
        self.assertEqual("62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                          smith.checksum_str("bar"))
        self.assertEqual("ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4",
                          smith.checksum_str(5))

    def test_checksum_file_exception(self):
        with self.assertRaises(smith.InternalException):
            smith.checksum_file("NOT A PATH")

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
