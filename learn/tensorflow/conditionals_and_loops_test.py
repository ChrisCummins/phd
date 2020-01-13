"""Learning conditional and loop patterns for TensorFlow."""
import typing

from labm8.py import app
from labm8.py import test
from third_party.py.tensorflow import tf

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


def ConditionalLessThan(x, y: int):
  return tf.cond(
    tf.less(x, tf.constant(y)),
    lambda: tf.constant(True),
    lambda: tf.constant(False),
  )


def test_ConditionalLessThan_interger_input():
  """Test conditional with simple lambdas and int input."""
  with tf.compat.v1.Session() as sess:
    assert sess.run(ConditionalLessThan(1, 10))
    assert not sess.run(ConditionalLessThan(10, 1))


def test_ConditionalLessThan_placeholder_input():
  """Test conditional with simple lambdas and placeholder input."""
  x = tf.compat.v1.placeholder(tf.int32, shape=())
  with tf.compat.v1.Session() as sess:
    assert sess.run(ConditionalLessThan(x, 10), feed_dict={x: 1})
    assert not sess.run(ConditionalLessThan(x, 1), feed_dict={x: 10})


def WhileLoopFactorial(n: int):
  """Calculating factorial in a while loop."""

  # The loop variables.
  class LoopVars(typing.NamedTuple):
    i: tf.Tensor
    imax: tf.Tensor
    acc: tf.Tensor

  def Condition(vars: LoopVars):
    """The while loop condition."""
    return tf.less(vars.i, vars.imax)

  def Body(vars: LoopVars):
    """The while loop body."""
    return [
      LoopVars(
        i=tf.add(vars.i, 1), imax=vars.imax, acc=tf.multiply(vars.acc, vars.i)
      ),
    ]

  init_vars = LoopVars(i=1, imax=n + 1, acc=1)
  while_loop = tf.while_loop(Condition, Body, [init_vars,])
  return while_loop


def test_WhileLoopFactorial():
  """Test factorial calculation."""
  with tf.compat.v1.Session() as sess:
    (final_vars,) = sess.run(WhileLoopFactorial(9))
    assert final_vars.acc == 362880  # 9!

    (final_vars,) = sess.run(WhileLoopFactorial(10))
    assert final_vars.acc == 3628800  # 10!


def WhileLoopFactorialIsEqualTo(x: int, y: int):
  """Combining the previous two functions, compute and return x! < y."""
  # Use the while loop output as input to the conditional.
  (final_vars,) = WhileLoopFactorial(x)

  cond = tf.cond(
    tf.equal(final_vars.acc, tf.constant(y)),
    lambda: tf.constant(True),
    lambda: tf.constant(False),
  )
  return cond


def test_WhileLoopFactorialIsLessThan():
  """Test while loop conditional combination."""
  with tf.compat.v1.Session() as sess:
    assert sess.run(WhileLoopFactorialIsEqualTo(1, 1))  # 1! = 1
    assert sess.run(WhileLoopFactorialIsEqualTo(2, 2))  # 2! = 2
    assert sess.run(WhileLoopFactorialIsEqualTo(3, 6))  # 3! = 6
    assert sess.run(WhileLoopFactorialIsEqualTo(4, 24))  # 4! = 24

    assert not sess.run(WhileLoopFactorialIsEqualTo(1, 2))  # 1! = 1
    assert not sess.run(WhileLoopFactorialIsEqualTo(2, 3))  # 2! = 2
    assert not sess.run(WhileLoopFactorialIsEqualTo(3, 7))  # 2! = 2
    assert not sess.run(WhileLoopFactorialIsEqualTo(4, 25))  # 4! = 24


if __name__ == "__main__":
  test.Main()
