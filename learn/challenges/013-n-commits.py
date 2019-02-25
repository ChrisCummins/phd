#!/usr/bin/env python3
#
# Challenge: Plot the distribution of commit message lengths in each
# of my public GitHub repos.
#
import json
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from github import Github

sns.set(color_codes=True)


def get_github_token(username, password):
  import requests
  response = requests.post(
      'https://api.github.com/authorizations',
      auth=(username, password),
      data=json.dumps({
          'scopes': ['repo'],
          'note': 'phd auth'
      }),
  )
  return response.json()['token']


def main():
  try:
    github_username = os.environ['GITHUB_USERNAME']
    github_pw = os.environ['GITHUB_PW']
    # github_token = get_github_token(github_username, github_pw)
  except KeyError as e:
    print('fatal: environment variable {} not set'.format(e))
    sys.exit(1)

  g = Github(github_username, github_pw)

  # Get sorted list of public, non-forked repos
  repos = [x for x in g.get_user().get_repos() if not x.fork and not x.private]
  repos.sort(key=lambda x: x.name.lower())

  # Read cache
  cache_path = __file__ + ".cache.json"
  if os.path.exists(cache_path):
    with open(cache_path, 'r') as infile:
      cache = json.load(infile)
  else:
    cache = {}

  # Get repo stats.
  stats = []
  for repo in repos:
    if repo.name == "clutter-android":  # FIXME: remove this hack!
      continue
    # Check cache first. Use last-updated timestamp to check for
    # updates.
    if (repo.name in cache and
        cache[repo.name]['updated_at'] == str(repo.updated_at)):
      print('Loading', repo.name, 'from cache')
      if len(cache[repo.name]['lc']):
        stats.append(cache[repo.name])
    else:
      print("Fetching", repo.name)
      lc = []
      wc = []

      for commit in repo.get_commits():
        # Only count commits that I wrote
        if commit.commit.author.email == github_username:
          message = commit.commit.message
          lc.append(len(message.split('\n')))
          wc.append(len(message.split()))

      data = {
          'name': repo.name,
          'updated_at': str(repo.updated_at),
          'lc': lc,
          'wc': wc
      }
      # update cache
      cache[repo.name] = data
      with open(cache_path, 'w') as outfile:
        json.dump(cache, outfile)

      # Check that there was at least one commit:
      if len(lc):
        stats.append(data)

  for stat in stats:
    grid = sns.distplot(stat['wc'])
  grid.set(xscale='log')
  plt.show()


if __name__ == '__main__':
  main()
