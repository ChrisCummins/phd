import dbus.service

from labm8.py import fs

LOCAL_DIR = fs.path("~/.omnitune")


class Error(Exception):
  pass


class Server(dbus.service.Object):
  pass
