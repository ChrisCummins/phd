from labm8 import app


# Univariate linear regression using iterative batch gradient descent.
#
def target_func(x):
  """function to estimate"""
  return 1.5 + 10.24 * x


def regression(data):
  """univariate linear regression. data is a sequence of tuples"""

  def cost(h0, h1, data):
    """cost function of h for data"""
    return sum([(h0 + h1 * d[0] - d[1])**2 for d in data]) / 2 * len(data)

  def gradient(h, a, data):
    """compute gradients of h with learning rate a for data"""
    g0 = sum([h[0] + h[1] * d[0] - d[1] for d in data]) / len(data)
    g1 = sum([(h[0] + h[1] * d[0] - d[1]) * d[0] for d in data]) / len(data)

    return g0, g1

  # Initial guess.
  h = [0, 0]
  # Learning rate.
  a = 0.05

  # Initial cost.
  j = cost(*h, data=data)

  # Threshold to stop iterations.
  j0 = 0.0000001

  # Iteration counter.
  i = 1

  while j > j0 and i < 1e6:
    # Compute gradients.
    g0, g1 = gradient(h, a, data)

    # Update gradients.
    h[0] -= a * g0
    h[1] -= a * g1

    # Update cost.
    j = cost(*h, data=data)

    # print("i = {i}, h = ({h0:.3f}, {h1:.3f}), J = {j:.3f}, "
    #       "g = ({g0:.3f}, {g1:.3f})"
    #       .format(i=i, h0=h[0], h1=h[1], j=j, g0=g0, g1=g1))
    i += 1

  print("y = {m:.3f}x + {c:.3f}".format(m=h[1], c=h[0]))
  return lambda x: h[0] + h[1] * x


def validate_hypothesis(h, data):
  for x, y in data:
    assert (abs(h(x) == y) < 0.0001)


def main(argv):
  del argv

  training_data1a = [(i, target_func(i)) for i in range(10)]
  h = regression(training_data1a)
  validate_hypothesis(h, training_data1a)


if __name__ == "__main__":
  app.RunWithArgs(main)
