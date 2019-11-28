import os


def cldrive_cli(
  platform: str, device: str, *args, timeout=os.environ.get("TIMEOUT", 60)
) -> str:
  """ get cldrive command """
  return [
    "cldrive",
    "-p",
    platform,
    "-d",
    device,
    "--debug",
    "--profiling",
    "-t",
    str(timeout),
    "-b",
  ] + list(args)
