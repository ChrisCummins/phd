"""Photolib main entry point."""
import subprocess
import sys

from labm8.py import app

if __name__ == "__main__":
  """Main entry point."""
  if len(sys.argv) == 1:
    raise app.UsageError("Usage: photolib <command> [args...}")

  command = sys.argv[1]
  if command == "--version":
    print(app.GetVersionInformationString())
    sys.exit(0)

  try:
    process = subprocess.Popen([f"photolib-{command}"] + sys.argv[2:])
    process.communicate()
    sys.exit(process.returncode)
  except FileNotFoundError:
    print(f"fatal: invalid command '{command}'")
    sys.exit(1)
  except KeyboardInterrupt:
    print(f"\r\033[Kkeyboard interrupt")
    sys.exit(1)
