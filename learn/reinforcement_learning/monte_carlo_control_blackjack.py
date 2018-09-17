"""Monte Carlo Control for the game of Blackjack.

Uses the OpenAI Gym Blackjack environment.
"""
import collections
import gym
import numpy as np
import typing
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_episodes', 10,
    'The number of episodes to run.')
flags.DEFINE_bool(
    'casino_blackjack_reward', True,
    'If True, the reward for a natural hand (an Ace and a 10 or face card) is '
    '1.5. Else the reward for the natural hand is the same as for a winning '
    'hand, 1.0.')

# The observation space is player score, dealer score, and whether or not there
# is a usable ace in the player's hand. A usable ace means that the sum of the
# hand with an Ace value of 11 is <= 21.
observation_t = typing.Tuple[int, int, bool]
Observation = collections.namedtuple(
    'Observation', ['player_score', 'dealer_score', 'usable_ace'])

# The action space is a single bool, either True to hit, or False to stick.
action_t = bool

# A single step includes the observation, the selected action, and its reward.
step_t = typing.Tuple[Observation, action_t, float]
Step = collections.namedtuple(
    'Step', ['observation', 'action', 'reward'])


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
    # Value table. The dimensions are: [player_score, dealer_score, usable_ace]
    self.N = np.zeros([21, 10, 2], dtype=np.int32)
    self.S = np.zeros([21, 10, 2], dtype=np.float)
    self.V = np.zeros([21, 10, 2], dtype=np.float)
    # The total number of episodes.
    self.num_episodes = 0

  def Reset(self):
    """Reset the internal state."""
    self.N.fill(0)
    self.S.fill(0)
    self.V.fill(0)
    self.num_episodes = 0

  def GetAnEpisode(self) -> typing.List[Step]:
    """Run an episode.

    The first step is the game opening. The action and reward are both None.

    Returns:
      A list of Step tuples.
    """
    done = False
    steps = [Step(Observation(*self.environment.reset()), None, None)]
    while not done:
      action = self.policy(steps[-1].observation)
      obs_, reward_, done, _ = self.environment.step(action)
      steps.append(Step(Observation(*obs_), action, reward_))
    return steps

  def Run(self, n: int):
    for i in range(n):
      self.num_episodes += 1
      episode = self.GetAnEpisode()
      logging.debug(
          'Episode %d, steps = %d, final_score = %02d:%02d, reward = %.1f',
          i, len(episode), episode[-1].observation.player_score,
          episode[-1].observation.dealer_score, episode[-1].reward)
      for j in range(1, len(episode)):
        step = episode[j]
        indices = (step.observation.player_score - 1,
                   step.observation.dealer_score - 1,
                   1 if step.observation.usable_ace else 0)
        if j < len(episode) - 1:
          self.N[indices] += 1
          self.S[indices] += sum(
              episode[k].reward for k in range(j, len(episode)))
          self.V[indices] = self.S[indices] / self.N[indices]


if __name__ == '__main__':
  app.run(main)
