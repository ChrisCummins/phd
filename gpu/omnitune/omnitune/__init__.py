import dbus.service

from phd.lib.labm8 import fs


LOCAL_DIR = fs.path("~/.omnitune")


class Error(Exception):
  pass


class Server(dbus.service.Object):
  pass
