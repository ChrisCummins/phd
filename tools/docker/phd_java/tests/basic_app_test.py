"""An example app to run in a docker image."""
from labm8.py import app

FLAGS = app.FLAGS


def main():
  """Main entry point."""
  app.Log(1, "Tests pass.")


if __name__ == "__main__":
  app.Run(main)
