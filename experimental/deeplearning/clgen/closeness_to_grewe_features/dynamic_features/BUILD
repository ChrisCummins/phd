#from absl import app
#from absl import flags
#from absl import logging
#
#from experimental.deeplearning.clgen.closeness_to_grewe_features import \
#  grewe_features_db
#from gpu.cldrive import env as cldrive_env
#from labm8 import system
#from labm8 import text
#from research.cummins_2017_cgo import opencl_kernel_driver

py_binary(
    name = "drive_with_cldrive",
    srcs = ["drive_with_cldrive.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//gpu/cldrive:env",
        "//labm8:system",
        "//labm8:text",
        "//research/cummins_2017_cgo:opencl_kernel_driver",
        "//third_party/py/absl",
    ],
)
