import dbus


def bytes2str(data):
    """
    Convert a sequence of bytes to a python string.
    """
    return "".join([chr(byte) for byte in data])


def parse_str(msg):
    """
    Parse Dbus string argument.
    """
    if isinstance(msg[0], dbus.Byte):
        return bytes2str(msg)
    else:
        return msg
