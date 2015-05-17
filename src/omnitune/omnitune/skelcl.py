import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

import labm8
from labm8 import crypto
from labm8 import io

import omnitune
from omnitune import util

SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"


class SkelCLProxy(omnitune.Proxy):

    @dbus.service.method(INTERFACE_NAME, in_signature='ss', out_signature='(nn)')
    def RequestWorkgroupSize(self, device_name, source):
        wg = (64, 32)

        device_name = util.parse_str(device_name).strip()
        source = util.parse_str(source)
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
