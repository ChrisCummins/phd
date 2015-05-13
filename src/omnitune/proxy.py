#!/usr/bin/env python2

from __future__ import print_function

import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

SESSION_NAME = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"

class SkelCLProxy(dbus.service.Object):

    @dbus.service.method(INTERFACE_NAME, in_signature='', out_signature='(nn)')
    def RequestWorkgroupSize(self):
        wg = (64, 32)
        print("RequestWorkGroupSize() -> ({0}, {1})".format(*wg))
        return wg

    @dbus.service.method(INTERFACE_NAME,
                         in_signature='', out_signature='')
    def Exit(self):
        mainloop.quit()

def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SessionBus()
    name = dbus.service.BusName(SESSION_NAME, bus)
    print("Launched session %s ..." % SESSION_NAME)

    # Launch SkelCL proxy.
    SkelCLProxy(bus, "/SkelCLProxy")
    print("Registered object %s/SkelCLProxy ..." % SESSION_NAME)

    mainloop = gobject.MainLoop()
    mainloop.run()

if __name__ == "__main__":
    main()
