# Experiments in Random Optimization

This is an [OpenAI Gym](https://gym.openai.com) environment for the LLVM
optimizer.

Example usage:

```py
env = gym.make(FLAGS.env)
env.reset()
for _ in range(10):
  # We don't yet have an observation space, so all we can do is take random
  # choices.
  obs, reward, done, info = env.step(env.action_space.sample())
  env.render()
  if done:
    break
```

Example run:

```sh
$ bazel run //experimental/compilers/random_opt -- \
    --env=LLVM-bzip2-512K-v0 --num_episodes=10 --max_steps=100
I0807 21:59:12.404618 140735642952576 random_opt.py:62] Generating environment LLVM-bzip2-512K-v0 ...
I0807 21:59:14.580255 140735642952576 random_opt.py:68] EPISODE 1:
==================================================
STEP #0

  Passes Run: [].
  Binary Runtimes: [395, 371, 388] ms.
  Reward: 0.000 (0.000 total)
  Speedup: 1.00x (1.00x total)

==================================================
STEP #1

  Passes Run: ['-view-regions'].
  Binary Runtimes: [4629, 1483, 2457] ms.
  Reward: -0.865 (-0.865 total)
  Speedup: 0.13x (0.13x total)

==================================================
STEP #2

  Passes Run: ['-structurizecfg'].
  Binary Runtimes: [1235, 585, 783] ms.
  Reward: 2.292 (1.427 total)
  Speedup: 3.29x (0.44x total)

...

==================================================
STEP #9

  Passes Run: ['-aarch64-simdinstr-opt'].
  Binary Runtimes: [] ms.
  Reward: -10.000 (-7.487 total)
  Speedup: 0.00x (0.00x total)
  Status: OPT_FAILED
```
