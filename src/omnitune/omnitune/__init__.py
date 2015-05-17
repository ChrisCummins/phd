import dbus
import dbus.service


class Error(Exception):
    pass


class Proxy(dbus.service.Object):
    pass
