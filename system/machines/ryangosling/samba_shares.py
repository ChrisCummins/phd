"""Generate samba share configs for ryangosling.

The ryangosling machine has a number of libraries that are exposed as samba
shares. I use two shares for each library, a browseable, read-only share, and
a hidden writable share.

This script generates:
  * The /etc/samba/smb.conf snippet which defines the shares.
  * A set of mkdir commands to generate mountpoints on a remote machine for
    these shares.
  * A list of fstab mounts for the samba shares.
"""
import pathlib
from typing import List
from typing import NamedTuple

from labm8.py import app


class NasShare(NamedTuple):
  """A samba share."""

  share_name: str
  relpath: str
  valid_users: List[str]


# A list of samba shares.
NAS_SHARES = [
  NasShare(
    share_name="Audio Projects", relpath="audio/projects", valid_users=["cec"]
  ),
  NasShare(
    share_name="Music Library", relpath="audio/third_party", valid_users=["cec"]
  ),
  NasShare(share_name="Binaries", relpath="bin", valid_users=["cec"]),
  NasShare(share_name="Documents", relpath="docs", valid_users=["cec"]),
  NasShare(
    share_name="Photo Library", relpath="img/photos", valid_users=["cec"]
  ),
  NasShare(share_name="Git Repositories", relpath="git", valid_users=["cec"]),
  NasShare(
    share_name="Comedians",
    relpath="video/third_party/comedians",
    valid_users=["cec"],
  ),
  NasShare(
    share_name="Movies", relpath="video/third_party/movies", valid_users=["cec"]
  ),
  NasShare(
    share_name="TV Shows", relpath="video/third_party/tv", valid_users=["cec"]
  ),
  NasShare(
    share_name="Video Projects", relpath="video/projects", valid_users=["cec"]
  ),
]

# The root path for share directories on ryangosling.
RYAN_GOSLING_LIBRARY_ROOT = pathlib.Path("~").expanduser()

# The root path for share mount points on remote machines.
MOUNT_POINT_ROOT = pathlib.Path("/mnt/ryangosling")

# The path to a file containing
CREDENTIALS_FILE = pathlib.Path(
  "~/.ryangosling_samba_credentials.txt"
).expanduser()

# The subdirectory names for non-writable and writable shares.
WRITABLE_DIRECTORY_NAMES = {False: "read_only", True: "writable"}


def GenerateSambaConfig(share: NasShare, writable: bool):
  """Generate a samba config."""
  share_name = f"Writable {share.share_name}" if writable else share.share_name
  return f"""\
[{share_name}]
  path = {RYAN_GOSLING_LIBRARY_ROOT / share.relpath}
  browseable = {"no" if writable else "yes"}
  read only = {"no" if writable else "yes"}
  guest ok = no
  valid users = {" ".join(share.valid_users)}\
"""


def GetFstabMountpoint(share: NasShare, writable: bool) -> str:
  """Construct a fstab mount command."""
  share_name = f"Writable {share.share_name}" if writable else share.share_name
  escaped_share_name = "\\040".join(share_name.split(" "))
  return (
    f"//ryangosling/{escaped_share_name} "
    f"{MOUNT_POINT_ROOT}/{WRITABLE_DIRECTORY_NAMES[writable]}/{share.relpath} cifs "
    f"uid=0,credentials={CREDENTIALS_FILE},iocharset=utf8,noperm 0 0"
  )


def Main():
  print("=== /etc/samba/smb.conf for ryangosling ===\n")
  for share in NAS_SHARES:
    print(GenerateSambaConfig(share, writable=False))
    print()
    print(GenerateSambaConfig(share, writable=True))
    print()

  print("\n=== Create mount points for remote machines ===\n")
  for share in NAS_SHARES:
    print(
      f"sudo mkdir -pv {MOUNT_POINT_ROOT}/{WRITABLE_DIRECTORY_NAMES[False]}/{share.relpath}"
    )
  for share in NAS_SHARES:
    print(
      f"sudo mkdir -pv {MOUNT_POINT_ROOT}/{WRITABLE_DIRECTORY_NAMES[True]}/{share.relpath}"
    )

  print("\n=== /etc/fstab mounts for remote machines ===\n")
  for share in NAS_SHARES:
    print(GetFstabMountpoint(share, writable=False))
  for share in NAS_SHARES:
    print(GetFstabMountpoint(share, writable=True))


if __name__ == "__main__":
  app.Run(Main)
