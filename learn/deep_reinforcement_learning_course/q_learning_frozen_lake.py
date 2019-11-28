"""Q learning for the game "Frozen Lake".

See: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
"""
import random
import typing

import gym
import numpy as np

from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS

app.DEFINE_integer('q_learning_seed', None,
                   'Random seed to initialize Q learning.')
app.DEFINE_integer('total_training_episodes', 10000,
                   'The total number of episodes to train for.')
app.DEFINE_float('q_learning_rate', 0.8, 'Initial Q learning rate.')
app.DEFINE_integer('max_steps_per_episode', 99,
                   'The maximum steps per episode.')
app.DEFINE_float('q_learning_gamma', 0.95, 'Q learning discounting rate.')
app.DEFINE_float('q_learning_init_epsilon', 1.0,
                 'Initial Q learning exploration rate.')
app.DEFINE_float('q_learning_min_epsilon', 0.01,
                 'Minimum Q learning exploration rate.')
app.DEFINE_float('q_learning_decay_rate', 0.01, 'The epsilon decay rate.')


def Train(env: gym.Env,
          total_episodes: int = 10000,
          max_steps: int = 99,
          learning_rate: float = 0.8,
          gamma: float = 0.95,
          init_epsilon: float = 1.0,
          min_epsilon: float = 0.01,
          decay_rate: float = 0.01,
          seed: typing.Optional[int] = None) -> np.ndarray:
  """Main entry point.

  Args:
    env: The gym environment.
    total_episodes: The total number of episodes to train.
    max_steps: The maximum number of steps per episode.
    learning_rate: The initial learning rate.
    gamma: The discounting rate.
    init_epsilon: The starting epsilon.
    min_epsilon: The minimum expsilon.
    decay_rate: Exponential decay rate for exploration epsilon.
    seed: An optional seed to set before training.

  Returns:
    The Q table.
  """
  if seed is None:
    seed = random.randint(0, 1000000000)

  np.random.seed(seed)
  random.seed(seed)
  app.Log(1, 'Random seed: %d.', seed)
  app.Log(
      1, 'Beginning training for %s episodes (max %s steps per episode). '
      'Initial learning rate: %.3f, decay rate: %.3f, '
      'initial epsilon: %.3f, min learning rate: %.3f, gamma: %.3f.',
      humanize.Commas(total_episodes), humanize.Commas(max_steps),
      learning_rate, decay_rate, init_epsilon, min_epsilon, gamma)

  app.Log(1, 'State space size: %s, action space size: %s. Q table: %dx%d.',
          humanize.Commas(env.observation_space.n),
          humanize.Commas(env.action_space.n), env.observation_space.n,
          env.action_space.n)

  q_table = np.zeros((env.observation_space.n, env.action_space.n))
  epsilon = init_epsilon
  rewards = []

  for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    total_rewards = 0

    for step in range(max_steps):
      # Exploration or exploitation.
      if random.uniform(0, 1) > epsilon:
        # Exploitation: Take the biggest Q value for this state.
        action = np.argmax(q_table[state, :])
      else:
        # Exploration: Take a random choice.
        action = env.action_space.sample()

      # Take the action (a) and observe the outcome state(s') and reward (r).
      new_state, reward, done, info = env.step(action)

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      # qtable[new_state,:] : all the actions we can take from new state
      q_table[state, action] = q_table[state, action] + learning_rate * (
          reward + gamma * np.max(q_table[new_state, :]) -
          q_table[state, action])

      total_rewards += reward
      state = new_state
      if done:  # Finish episode if we're done.
        break

    episode += 1
    # Reduce epsilon (because we need less and less exploration).
    epsilon = min_epsilon + (init_epsilon - min_epsilon) * np.exp(
        -decay_rate * episode)
    rewards.append(total_rewards)

  app.Log(1, 'Score over time: %.3f.', sum(rewards) / total_episodes)
  return q_table


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  print(
      Train(gym.make("FrozenLake-v0"),
            total_episodes=FLAGS.total_training_episodes,
            learning_rate=FLAGS.q_learning_rate,
            max_steps=FLAGS.max_steps_per_episode,
            gamma=FLAGS.q_learning_gamma,
            init_epsilon=FLAGS.q_learning_init_epsilon,
            min_epsilon=FLAGS.q_learning_min_epsilon,
            decay_rate=FLAGS.q_learning_decay_rate,
            seed=FLAGS.q_learning_seed))


if __name__ == '__main__':
  app.RunWithArgs(main)
