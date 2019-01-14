# An interface to Arduino hardware.

load("@platformio_rules//platformio:platformio.bzl", "platformio_library")

cc_library(
    name = "arduino_interface",
    hdrs = ["arduino_interface.h"],
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":digital_value",
        ":pin_mode",
    ],
)

platformio_library(
    name = "Arduino_interface",
    hdr = "arduino_interface.h",
    visibility = ["//maker/arduino:__subpackages__"],
    deps = [
        ":Arduino_interface_digital_value",
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
