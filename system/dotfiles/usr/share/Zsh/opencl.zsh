# OpenCL doesn't provide a means to temporarily enable or disable platforms, so
# these functions provide that ability my moving ICD files in and out of the
# global OpenCL vendors registry.
#
# Usage:
#   $ opencl_platforms_list                # list platforms
#   $ opencl_platforms_enable <platform>   # enable a platform
#   $ opencl_platforms_disable <platform>  # disable a platform

opencl_platforms_list() {
  test -d /etc/OpenCL/vendors || { echo "fatal: /etc/OpenCL/vendors not found"; return; }

  echo "Enabled:"
  ls /etc/OpenCL/vendors | sed 's/^/  /' | sed 's/\.icd$//'
  echo
  echo "Disabled:"
  ls /etc/OpenCL/vendors.disabled | sed 's/^/  /' | sed 's/\.icd$//'
}

opencl_platforms_enable() {
  test -d /etc/OpenCL/vendors || { echo "fatal: /etc/OpenCL/vendors not found"; return; }

  sudo mkdir -p /etc/OpenCL/vendors.disabled
  if [[ -z "$1" ]]; then
    echo "Disabled:"
    ls /etc/OpenCL/vendors.disabled | sed 's/^/  /' | sed 's/\.icd$//'
  else
    sudo mv -v /etc/OpenCL/vendors.disabled/"$1".icd /etc/OpenCL/vendors/"$1".icd
  fi
}

opencl_platforms_disable() {
  test -d /etc/OpenCL/vendors || { echo "fatal: /etc/OpenCL/vendors not found"; return; }

  if [[ -z "$1" ]]; then
    echo "Enabled:"
    ls /etc/OpenCL/vendors | sed 's/^/  /' | sed 's/\.icd$//'
  else
    sudo mkdir -p /etc/OpenCL/vendors.disabled
    sudo mv -v /etc/OpenCL/vendors/"$1".icd /etc/OpenCL/vendors.disabled/"$1".icd
  fi
}
