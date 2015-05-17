#!/usr/bin/env python2

from __future__ import print_function

import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

import labm8
from labm8 import crypto
from labm8 import io

SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"

class SkelCLProxy(dbus.service.Object):

    @staticmethod
    def parse_bytes(data):
        return ''.join([chr(byte) for byte in data])

    @staticmethod
    def parse_str(msg):
        if isinstance(msg[0], dbus.Byte):
            return SkelCLProxy.parse_bytes(msg)
        else:
            return msg

    @dbus.service.method(INTERFACE_NAME, in_signature='ss', out_signature='(nn)')
    def RequestWorkgroupSize(self, device_name, source):
        wg = (64, 32)

        device_name = self.parse_str(device_name).strip()
        source = self.parse_str(source)
        source_id = crypto.sha1(source)

        io.debug(("RequestWorkGroupSize({dev}, {id}) -> ({c}, {r})"
                  .format(dev=device_name[:8], id=source_id[:8],
                          c=wg[0], r=wg[1])))
        return wg

    @dbus.service.method(INTERFACE_NAME, in_signature='', out_signature='')
    def Exit(self):
        mainloop.quit()

def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SessionBus()
    name = dbus.service.BusName(SESSION_NAME, bus)
    io.info("Launched session %s ..." % SESSION_NAME)

    # Launch SkelCL proxy.
    SkelCLProxy(bus, "/SkelCLProxy")
    io.info("Registered object %s/SkelCLProxy ..." % SESSION_NAME)

    mainloop = gobject.MainLoop()
    mainloop.run()

if __name__ == "__main__":
    main()
