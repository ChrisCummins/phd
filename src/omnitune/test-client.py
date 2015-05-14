#!/usr/bin/env python2

from __future__ import print_function

import timeit
import sys

import time

import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

SESSION_NAME = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"

wgs = []
def test_RequestWgSize(proxy):
    wg = proxy.RequestWorkgroupSize(["foo", "bar"])

    wgs.append((int(wg[0]), int(wg[1])))

def main():
    try:
        bus = dbus.SessionBus()
        session = bus.get_object(SESSION_NAME, "/SkelCLProxy")
        print("Connected to session %s ..." % SESSION_NAME)
        proxy = dbus.Interface(session, INTERFACE_NAME)
        print("Connected to interface %s ..." % INTERFACE_NAME)

        start = time.time()
        n = 5
        for i in range(n):
            test_RequestWgSize(proxy)
        end = time.time()
        print(n, "requests in", end - start, "seconds")
        print("last value:", wgs[-1])

    except dbus.DBusException as err:
        print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
