import json
from os import path

import requests
import rumps
from pync import Notifier


# Read global configuration file
with open(path.expanduser("~/.config/toggl.json")) as infile:
  config = json.load(infile)

# Sanity check config
assert "auth" in config
assert "workspace" in config

# Create auth
auth = requests.auth.HTTPBasicAuth(config["auth"], "api_token")


class TimerSymbol(object):
  INACTIVE = "🚫"
  ACTIVE = "⏰"


def assert_or_fail(*args, **kwargs):
  # TODO:
  pass


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


def get_active_timer() -> str:
  # FIXME:
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


class TimerMenubar(rumps.App):

  @rumps.clicked("Update Title")
  def update_title(self, _):
    self.title = get_active_timer()

  @rumps.clicked("Test Alert")
  def test_alert(self, _):
    rumps.alert("Alert!")

  @rumps.clicked("Test Notification")
  def test_notification(self, _):
    Notifier.notify('Test Notification', title='Timer')

  @rumps.clicked("Test Checkbox")
  def test_button(self, sender):
    sender.state = not sender.state


if __name__ == "__main__":
  TimerMenubar(TimerSymbol.INACTIVE).run()
