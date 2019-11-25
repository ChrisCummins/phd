"""Monte Carlo Control for the game of Blackjack.

Uses the OpenAI Gym Blackjack environment.
"""
import collections
import typing

import gym
import numpy as np

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_integer('num_episodes', 10000, 'The number of episodes to run.')
app.DEFINE_boolean(
    'casino_blackjack_reward', True,
    'If True, the reward for a natural hand (an Ace and a 10 or face card) is '
    '1.5. Else the reward for the natural hand is the same as for a winning '
    'hand, 1.0.')

# The observation space is player score, dealer score, and whether or not there
# is a usable ace in the player's hand. A usable ace means that the sum of the
# hand with an Ace value of 11 is <= 21.
observation_t = typing.Tuple[int, int, bool]


class Observation(typing.NamedTuple):
  player_score: int
  dealer_score: int
  usable_ace: bool


# The action space is a single bool, either True to hit, or False to stick.
action_t = bool

# A single step includes the observation, the selected action, and its reward.
step_t = typing.Tuple[Observation, action_t, float]


class Step(typing.NamedTuple):
  observation: typing.Any
  action: typing.Any
  reward: float


class MonteCarloControlBlackjack(object):
  """A Monte Carlo Control "agent" for the game of Blackjack."""

  def __init__(self, casino_blackjack_reward: bool,
               policy: typing.Callable[[Observation], action_t]):
    """Create the agent.

    Args:
      casino_blackjack_reward: If True, the reward for a natural blackjack is
        1.5.
      policy: A callback which accepts an Observation and returns True to Hit,
        or False to stick.
    """
    self.environment = gym.make('Blackjack-v0')
    self.environment.natural = casino_blackjack_reward
    self.policy = policy
    # Table dimensions are:
    #   [player_score, dealer_score, usable_ace, action]
    self.N = np.zeros([21, 10, 2, 2], dtype=np.int32)
    self.S = np.zeros([21, 10, 2, 2], dtype=np.float)
    self.Q = np.zeros([21, 10, 2, 2], dtype=np.float)
    # Table dimensions are:
    #   [player_score, dealer_score, usable_ace]
    self.pi = np.zeros([21, 10, 2], dtype=np.bool)
    # The total number of episodes.
    self.num_episodes = 0
    self.num_wins = 0
    self.num_losses = 0

  def Reset(self):
    """Reset the internal state."""
    self.N.fill(0)
    self.S.fill(0)
    self.Q.fill(0)
    self.pi.fill(False)
    self.num_episodes = 0
    self.num_wins = 0
    self.num_losses = 0

  def GetAnEpisode(self) -> typing.List[Step]:
    """Run an episode.

    The first step is the game opening. The action and reward are both None.

    Returns:
      A list of Step tuples.
    """
    done = False
    steps = [Step(Observation(*self.environment.reset()), None, None)]
    started = False
    while not done:
      if started:
        action = self.pi[steps[-1].observation.player_score -
                         1, steps[-1].observation.dealer_score -
                         1, 1 if steps[-1].observation.usable_ace else 0,]
      else:
        action = np.random.randint(0, 2, dtype=np.bool)
        started = True
      # self.policy(steps[-1].observation)
      obs_, reward_, done, _ = self.environment.step(action)
      steps.append(Step(Observation(*obs_), action, reward_))
    return steps

  def Run(self, n: int):
    for i in range(n):
      self.num_episodes += 1
      episode = self.GetAnEpisode()
      app.Log(2,
              'Episode %d, steps = %d, final_score = %02d:%02d, reward = %.1f',
              i, len(episode), episode[-1].observation.player_score,
              episode[-1].observation.dealer_score, episode[-1].reward)
      for j in range(1, len(episode)):
        state = episode[j - 1].observation
        indices = (state.player_score - 1, state.dealer_score - 1,
                   1 if state.usable_ace else 0, 1 if episode[j].action else 0)
        self.N[indices] += 1
        self.S[indices] += sum(
            episode[k].reward for k in range(j, len(episode)))
        self.Q[indices] = self.S[indices] / self.N[indices]

      for j in range(0, len(episode) - 1):
        state = episode[j].observation
        indices = (state.player_score - 1, state.dealer_score - 1,
                   1 if state.usable_ace else 0)
        self.pi[indices] = (
            True if self.Q[(*indices, 1)] > self.Q[(*indices, 0)] else False)

      if episode[-1].reward > 0:
        self.num_wins += 1
      elif episode[-1].reward < 0:
        self.num_losses += 1

  @property
  def win_ratio(self) -> float:
    return self.num_wins / max(self.num_episodes, 1)


def main(argv):
  del argv
  agent = MonteCarloControlBlackjack(
      casino_blackjack_reward=FLAGS.casino_blackjack_reward,
      policy=lambda obs: obs.player_score < 17)
  agent.Run(FLAGS.num_episodes)
  print('Policy (no usable ace):')
  print(agent.pi[10:, :, 0])
  print('\nPolicy (usable ace):')
  print(agent.pi[10:, :, 1])
  print(f'After {agent.num_episodes} iterations, '
        f'win ratio {agent.win_ratio:.1%}')


if __name__ == '__main__':
  app.RunWithArgs(main)
