import logging


def LogShellCommand(*args):
  if logging.getLogger().level <= logging.INFO:
    app.Log(2, Colors.PURPLE + "    $ " + "".join(*args) + Colors.END)


def LogShellOutput(stdout):
  if logging.getLogger().level <= logging.INFO and len(stdout):
    indented = '    ' + '\n    '.join(stdout.rstrip().split('\n'))
    app.Log(2, Colors.YELLOW + indented + Colors.END)


class Colors(object):
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'


def SetVerbosity(verbose=False):
  log_level = logging.DEBUG if verbose else logging.INFO
  logging.basicConfig(level=log_level, format="%(message)s")


def Print(*msg, **kwargs):
  sep = kwargs.get("sep", " ")
  text = sep.join(msg)
  indented = "\n        > ".join(text.split("\n"))
  app.Log(1, io.Colors.GREEN + "        > " + indented + io.Colors.END)
