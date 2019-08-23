"""Photolib main entry point."""
import sys
import subprocess

from labm8 import app

if __name__ == '__main__':
  """Main entry point."""
  if len(sys.argv) == 1:
    raise app.UsageError("Usage: photolib <command> [args...}")

  command = sys.argv[1]
  if command == '--version':
    print(app.GetVersionInformationString())
    sys.exit(0)

  try:
    process = subprocess.Popen([f'photolib-{command}'])
    process.communicate()
    sys.exit(process.returncode)
  except FileNotFoundError:
    print(f"fatal: invalid command '{command}'")
    sys.exit(1)
