/*
 * client.cc - omnitune C++ client test file.
 */

// GLib DBus C++ interface.
#include <glibmm.h>
#include <giomm.h>
#include <giomm/dbusproxy.h>

// OpenCL headers.
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#undef __CL_ENABLE_EXCEPTIONS

// Integer types.
#include <cstdint>

#define BUS_NAME       "org.omnitune"
#define INTERFACE_NAME "org.omnitune.skelcl"
#define OBJECT_PATH    "/SkelCLProxy"

// Global variable for omnitune proxy.
Glib::RefPtr<Gio::DBus::Proxy> proxy;

// Request a workgroup size from the proxy server and set the response
// to local.
void requestWgSize(cl_uint *const local) {
    // Synchronously call "RequestWorkgroupSize()".
    Glib::VariantContainerBase response
            = proxy->call_sync("RequestWorkgroupSize");
    Glib::VariantIter iterator(response.get_child(0));
    Glib::Variant<int16_t> var;

    // Set response values.
    iterator.next_value(var);
    local[0] = var.get();
    iterator.next_value(var);
    local[1] = var.get();
}

int main() {
    Glib::init();
    Gio::init();

    Glib::RefPtr<Gio::DBus::Connection> bus
            = Gio::DBus::Connection::get_sync(Gio::DBus::BUS_TYPE_SESSION);
    proxy = Gio::DBus::Proxy::create_sync(bus,
                                     BUS_NAME,
                                     OBJECT_PATH,
                                     INTERFACE_NAME);

    printf("Start\n");
    for (int i = 0; i < 1000; i++) {
        cl_uint local[2];
        requestWgSize(&local[0]);
    }
    printf("Done\n");

    return 0;
}
