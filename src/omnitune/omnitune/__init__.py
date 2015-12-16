import dbus
import dbus.service

import labm8
from labm8 import fs

LOCAL_DIR = fs.path("~/.omnitune")


class Error(Exception):
    pass


class Server(dbus.service.Object):
    pass
