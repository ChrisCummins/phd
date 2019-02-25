#!/usr/bin/env python3.6

import datetime
import json
import re
import sys
from argparse import ArgumentParser
from os import path
from typing import List

import requests

# Read global configuration file
with open(path.expanduser("~/.config/toggl.json")) as infile:
  config = json.load(infile)

# Sanity check config
assert "auth" in config
assert "workspace" in config

# Create auth
auth = requests.auth.HTTPBasicAuth(config["auth"], "api_token")


class Colors:
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


class _getch_unix:

  def __init__(self):
    pass

  def __call__(self):
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class _getch_windows:

  def __init__(self):
    pass

  def __call__(self):
    import msvcrt
    return msvcrt.getch()


class getch:

  def __init__(self):
    try:
      self.impl = _getch_windows()
    except ImportError:
      self.impl = _getch_unix()

  def __call__(self):
    return self.impl()


def assert_or_fail(condition: bool, error_msg: str = "",
                   returncode: int = 1) -> None:
  if not condition:
    msg = f"fatal: {error_msg}" if len(error_msg) else "fatal error!"
    print(msg, file=sys.stderr)
    sys.exit(returncode)


def GET(*args, **kwargs):
  r = requests.get(*args, **kwargs, auth=auth)
  assert_or_fail(r.status_code == 200)
  return r


def PUT(*args, **kwargs):
  r = requests.put(*args, **kwargs, auth=auth)
  assert_or_fail(r.status_code == 200)
  return r


def POST(*args, **kwargs):
  r = requests.post(*args, **kwargs, auth=auth)
  assert_or_fail(r.status_code == 200)
  return r


def DELETE(*args, **kwargs):
  r = requests.delete(*args, **kwargs, auth=auth)
  assert_or_fail(r.status_code == 200)
  return r


def parse_date(string: str) -> datetime.datetime:
  pass


def get_active_timer() -> str:
  r = GET("https://www.toggl.com/api/v8/time_entries/current")
  timer_data = r.json()["data"]

  if timer_data:
    pid = timer_data["pid"]
    tag = timer_data["tags"][0]
    start_time = timer_data["start"]
    tag_name = tag
    return f"{start_time} ⏰ {tag}"
  else:
    return "🚫 No Timer"


def prompt_for_choice(choices: List[chr], default=None) -> chr:
  get_char = getch()
  while True:
    char = str(get_char()).lower()
    if char == '\x03':
      raise KeyboardInterrupt
    if char in choices:
      return char


def stop_timer() -> None:
  r = GET("https://www.toggl.com/api/v8/time_entries/current")
  timer_data = r.json()["data"]

  if timer_data:
    tid = timer_data["id"]
    PUT(f"https://www.toggl.com/api/v8/time_entries/{tid}/stop")


def start_timer(pid: int, tags: List[str]) -> None:
  stop_timer()
  payload = {
      "time_entry": {
          "pid": pid,
          "created_with": "Command Line",
          "tags": tags,
      }
  }
  POST(
      "https://www.toggl.com/api/v8/time_entries/start",
      headers={"Content-Type": "application/json"},
      data=json.dumps(payload))


def prompt_for_project():
  wid = config["workspace"]
  r = GET(f"https://www.toggl.com/api/v8/workspaces/{wid}/projects")

  projects = {}
  for data in r.json():
    if data["active"]:
      projects[data["name"]] = data["id"]

  choices = sorted(projects.keys()) + ["+ New"]

  print("\nSelect project")
  for i, choice in enumerate(choices):
    j = i + 1
    print(f"[{Colors.BOLD}{j}{Colors.END}] {choice}")

  choice = prompt_for_choice(str(i) for i in range(1, len(choices) + 1))
  choice_index = int(choice) - 1

  if choice_index == len(choices) - 1:
    project_name = input("Name new project: ").strip()

    project = {
        "name": project_name,
        "wid": config["workspace"],
        "is_private": False,
    }

    r = POST(
        "https://www.toggl.com/api/v8/projects",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"project": project}))
    pid = r.json()["data"]["id"]
  else:
    project_name = choices[choice_index]
    pid = projects[project_name]

  return {"name": project_name, "pid": pid}


def delete_active_timer():
  r = GET("https://www.toggl.com/api/v8/time_entries/current")
  timer_data = r.json()["data"]
  if timer_data:
    tid = timer_data["id"]
    DELETE(f"https://www.toggl.com/api/v8/time_entries/{tid}")


def prompt_for_tag(project_name: str):
  wid = config["workspace"]
  r = GET(f"https://www.toggl.com/api/v8/workspaces/{wid}/tags")

  choices = []
  for data in r.json():
    if data["name"].startswith(project_name + "/"):
      choices.append(re.sub(project_name + "/", "", data["name"]))
  choices = sorted(choices)
  choices.append("+ New")

  print("\nSelect tag")
  for i, choice in enumerate(choices):
    j = i + 1
    print(f"[{Colors.BOLD}{j}{Colors.END}] {choice}")

  choice = prompt_for_choice(str(i) for i in range(1, len(choices) + 1))
  choice_index = int(choice) - 1

  if choice_index == len(choices) - 1:
    tag_name = input("Name new tag: ").strip()
  else:
    tag_name = choices[choice_index]
  tag_name = project_name + "/" + tag_name

  return tag_name


def tui():
  timer = get_active_timer()

  choices = []
  if timer != "🚫 No Timer":
    choices += [
        (f"[{Colors.BOLD}S{Colors.END}]top timer", "s"),
        (f"[{Colors.BOLD}D{Colors.END}]elete timer", "d"),
    ]
  choices.append((f"[{Colors.BOLD}N{Colors.END}]ew timer", "n"))

  print(timer)
  print()
  print("Actions:", ", ".join(c[0] for c in choices))

  choice = prompt_for_choice(list("".join(c[1] for c in choices)))

  if choice == "s":
    stop_timer()
  elif choice == "d":
    delete_active_timer()
  elif choice == "n":
    project = prompt_for_project()
    tag = prompt_for_tag(project_name=project["name"])
    start_timer(pid=project["pid"], tags=[tag])
  else:
    raise NotImplementedError

  print(get_active_timer())


def main():
  try:
    parser = ArgumentParser()  # help="time tracking")
    parser.add_argument("--active-timer", action="store_true")
    args = parser.parse_args()

    if args.active_timer:
      print(get_active_timer())
    else:
      tui()
  except KeyboardInterrupt:
    sys.exit(1)


if __name__ == "__main__":
  main()
