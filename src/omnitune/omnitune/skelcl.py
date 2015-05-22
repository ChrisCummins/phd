import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

import labm8
from labm8 import crypto
from labm8 import fs
from labm8 import io

import omnitune
from omnitune import cache
from omnitune import util
from omnitune import llvm

SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"
OBJECT_PATH    = "/"


STENCIL_KERNEL_FEATURES = (
    "instructions (of all types)",
    "ratio AShr insts",
    "ratio Add insts",
    "ratio Alloca insts",
    "ratio And insts",
    "ratio Br insts",
    "ratio Call insts",
    "ratio FAdd insts",
    "ratio FCmp insts",
    "ratio FDiv insts",
    "ratio FMul insts",
    "ratio FPExt insts",
    "ratio FPToSI insts",
    "ratio FSub insts",
    "ratio GetElementPtr insts",
    "ratio ICmp insts",
    "ratio InsertValue insts",
    "ratio Load insts",
    "ratio Mul insts",
    "ratio Or insts",
    "ratio PHI insts",
    "ratio Ret insts",
    "ratio SDiv insts",
    "ratio SExt insts",
    "ratio SIToFP insts",
    "ratio SRem insts",
    "ratio Select insts",
    "ratio Shl insts",
    "ratio Store insts",
    "ratio Sub insts",
    "ratio Trunc insts",
    "ratio UDiv insts",
    "ratio Xor insts",
    "ratio ZExt insts",
    "ratio basic blocks",
    "ratio memory instructions",
    "ratio non-external functions"
)


def checksum_str(string):
    """
    Return the checksum for a string.
    """
    return crypto.sha1(string)


def vectorise_ratios(ratios):
    """
    Vectorise a dictionary of stencil kernel features.
    """
    vector = []
    for feature in STENCIL_KERNEL_FEATURES:
        if feature in ratios:
            vector.append(ratios[feature])
        else:
            vector.append(0)
    return vector


def get_source_features(source, path=""):
    bitcode = llvm.bitcode(source, path=path)
    instcounts = llvm.instcounts(bitcode, path=path)
    ratios = llvm.instcounts2ratios(instcounts)

    return vectorise_ratios(ratios)


def get_device_features(device_name):
    return []


class SkelCLProxy(omnitune.Proxy):

    LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")

    def __init__(self, *args, **kwargs):
        """
        Construct a SkelCL proxy.
        """
        super(SkelCLProxy, self).__init__(*args, **kwargs)
        io.info("Registered proxy %s/SkelCLProxy ..." % SESSION_NAME)
        self.kcache = cache.JsonCache("/tmp/omnitune-skelcl-kcache.json")
        self.dcache = cache.JsonCache("/tmp/omnitune-skelcl-dcache.json")

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiis', out_signature='(nn)')
    def RequestStencilParams(self, device_name, device_count,
                             north, south, east, west, data_width,
                             data_height, source):
        """
        Request a set of parameter values for a stencil skeleton.

        @param device_name OpenCL device name.
        @param source SkelCL stencil program source code.
        """

        # Parse arguments.
        device_name = util.parse_str(device_name).strip()
        device_count = int(device_count)
        north = int(north)
        south = int(south)
        east = int(east)
        west = int(west)
        data_width = int(data_width)
        data_height = int(data_height)
        source = util.parse_str(source)

        # Calculate source checksum.
        checksum = checksum_str(source)

        sourcefeatures = self.kcache.get(checksum)
        if sourcefeatures is None:
            features = get_source_features(source, path=self.LLVM_PATH)
            sourcefeatures = self.kcache.set(checksum, features)

        devicefeatures = self.dcache.get(device_name)
        if devicefeatures is None:
            features = get_device_features(device_name)
            devicefeatures = self.dcache.set(device_name, features)

        features = devicefeatures + sourcefeatures + [
            device_count,
            north, south, east, west,
            data_width, data_height
        ]
        print(features)

        wg = (64, 32)

        io.debug(("RequestStencilParams({dev}, {count}, "
                  "[{n}, {s}, {e}, {w}], {width}, {height}, {id}) ->"
                  "({c}, {r})"
                  .format(dev=device_name[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          id=checksum[:8],
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
    SkelCLProxy(bus, OBJECT_PATH)

    mainloop = gobject.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        labm8.exit()
