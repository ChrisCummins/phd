# An interface to Arduino hardware.

load("@platformio_rules//platformio:platformio.bzl", "platformio_library")

cc_library(
    name = "analog_value",
    srcs = ["analog_value.cc"],
    hdrs = ["analog_value.h"],
)

platformio_library(
    name = "Arduino_interface_analog_value",
    src = "analog_value.cc",
    hdr = "analog_value.h",
)

cc_library(
    name = "arduino_interface",
    hdrs = ["arduino_interface.h"],
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":analog_value",
        ":digital_value",
        ":pin",
        ":pin_mode",
    ],
)

platformio_library(
    name = "Arduino_interface",
    hdr = "arduino_interface.h",
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":Arduino_interface_analog_value",
        ":Arduino_interface_digital_value",
        ":Arduino_interface_pin",
        ":Arduino_interface_pin_mode",
    ],
)

cc_library(
    name = "mock_arduino_interface",
    testonly = 1,
    srcs = ["mock_arduino_interface.cc"],
    hdrs = ["mock_arduino_interface.h"],
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":arduino_interface",
        ":mock_liquid_crystal",
        "@gtest",
    ],
)

cc_library(
    name = "mock_liquid_crystal",
    testonly = 1,
    hdrs = ["mock_liquid_crystal.h"],
    deps = [
        "@gtest",
    ],
)

platformio_library(
    name = "Arduino_hardware",
    src = "arduino_interface_hardware.cc",
    hdr = "arduino_interface_hardware.h",
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [":Arduino_interface"],
)

# Digital value.

cc_library(
    name = "digital_value",
    srcs = ["digital_value_mock.cc"],
    hdrs = ["digital_value.h"],
)

platformio_library(
    name = "Arduino_interface_digital_value",
    src = "digital_value_hardware.cc",
    hdr = "digital_value.h",
)

# IP address.

cc_library(
    name = "ip_address",
    hdrs = ["ip_address.h"],
)

platformio_library(
    name = "Arduino_interface_ip_address",
    hdr = "ip_address.h",
)

# Pin.

cc_library(
    name = "pin",
    hdrs = ["pin.h"],
)

platformio_library(
    name = "Arduino_interface_pin",
    hdr = "pin.h",
)

# Pin mode.

cc_library(
    name = "pin_mode",
    srcs = ["pin_mode_mock.cc"],
    hdrs = ["pin_mode.h"],
)

platformio_library(
    name = "Arduino_interface_pin_mode",
    src = "pin_mode_hardware.cc",
    hdr = "pin_mode.h",
)

# WiFi server.

cc_library(
    name = "wifi_server",
    srcs = ["wifi_server_mock.cc"],
    hdrs = ["wifi_server.h"],
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":arduino_interface",
        ":ip_address",
        "@gtest",
    ],
)

platformio_library(
    name = "Arduino_interface_wifi_server",
    src = "wifi_server_hardware.cc",
    hdr = "wifi_server.h",
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":Arduino_interface",
        ":Arduino_interface_ip_address",
    ],
)
