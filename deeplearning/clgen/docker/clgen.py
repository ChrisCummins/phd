"""Main entry point for docker binary.

The job of this file is to simply call into the main CLgen binary, having
set some default arg values.
"""

from config import getconfig
from deeplearning.clgen import clgen
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_boolean("version", False,
                   "Print CLgen version information and exit.")


def main():
  if FLAGS.version:
    config = getconfig.GetGlobalConfig()
    print(f"""
CLgen docker ({config.uname}-{"gpu" if config.with_cuda else "cpu"}).

Made with \033[1;31m♥\033[0;0m by Chris Cummins <chrisc.101@gmail.com>.
https://chriscummins.cc/clgen
""".strip())
    return

  config = clgen.ConfigFromFlags()
  instance = clgen.Instance(config)
  with instance.Session():
    instance.Sample(
        min_num_samples=FLAGS.min_samples,
        print_samples=True,
        cache_samples=False)


if __name__ == '__main__':
  app.Run(main)
