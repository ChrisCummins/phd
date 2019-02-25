"""Generate code fragments with loops."""
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


class Context(object):

  def __init__(self):
    self.variable_counter = -1

  def NextVariableIdentifier(self):
    self.variable_counter += 1
    return str(chr(ord('a') + self.variable_counter))


class NamedVariable(object):

  def __init__(self):
    self.identifier = None

  def Finalize(self, ctx: Context):
    self.identifier = ctx.NextVariableIdentifier()
    return self

  def __repr__(self) -> str:
    return self.identifier


class Statement(object):

  def Finalize(self, ctx: Context):
    return self


class Loop(Statement):

  def __init__(self, max_iter: int, loop_body: Statement) -> Statement:
    self.iterator = NamedVariable()
    self.max_iter = max_iter
    self.loop_body = loop_body

  def __repr__(self) -> str:
    return '\n'.join([
        f'for (int {self.iterator} = 0; {self.iterator} < {self.max_iter}; ++{self.iterator}) {{',
        '\n'.join(f'  {l}' for l in str(self.loop_body).split('\n')),
        '}',
    ])

  def Finalize(self, ctx: Context):
    self.iterator.Finalize(ctx)
    self.loop_body.Finalize(ctx)
    return self


class DummyLoopBody(Statement):

  def __repr__(self) -> None:
    return 'A();'


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Hello, world!')
  print(Loop(100, DummyLoopBody()).Finalize(Context()))
  inner_loop = Loop(10, DummyLoopBody())
  print(Loop(10, inner_loop).Finalize(Context()))


if __name__ == '__main__':
  app.run(main)
