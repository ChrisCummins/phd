import collections
import logging
import socket
import sys

TASKS = {}


class SchedulingError(Exception):
  pass


class InvalidTaskError(Exception):
  pass


class DuplicateTaskName(InvalidTaskError):
  pass


class TaskArgumentError(InvalidTaskError):

  def __init__(self, argument, message):
    self.argument = argument
    self.message = message

  def __str__(self):
    return ' '.join([self.argument, self.message])


class Task(object):

  def __init__(self, name=None, genfiles=None, deps=None):
    print(self, "__init__")
    if name in TASKS:
      raise DuplicateTaskName(name)
    TASKS[name] = self
    self.name = AssertSet(name, 'name')
    self.genfiles = genfiles or []
    self.deps = deps or []

  def SetUp(self, ctx):
    pass

  def __call__(self, ctx):
    pass

  def TearDown(self, ctx):
    pass


from experimental.system.dotfiles.implementation.tasks import *


def IsRunnableTask(t):
  """Returns true if object is a task for the current platform.

  Args:
    t: A Task instance.

  Returns:
    True if task instance is runnable.
  """
  # Check that object is a class and inherits from 'Task':
  if not isinstance(t, Task):
    return False

  # Check that hostname is whitelisted (if whitelist is provided):
  hosts = getattr(t, "__hosts__", [])
  if hosts and socket.gethostname() not in hosts:
    msg = "skipping " + type(t).__name__ + " on host " + socket.gethostname()
    app.Log(2, msg)
    return False

  # Check that task passes all req tests:
  reqs = getattr(t, "__reqs__", [])
  if reqs and not all(req() for req in reqs):
    msg = "skipping " + type(t).__name__ + ", failed req check"
    app.Log(2, msg)
    return False

  return True


def ScheduleTask(task_name, schedule, all_tasks, depth=1):
  """ recursively schedule a task and its dependencies """
  # Sanity check for scheduling errors:
  if depth > 1000:
    raise SchedulingError("failed to resolve schedule for task '" + task_name +
                          "' after 1000 tries")
    sys.exit(1)

  # Instantiate the task class:
  if not task_name in all_tasks:
    raise SchedulingError("task '" + task_name + "' not found!")
  t = TASKS[task_name]

  # If the task is not runnable, schedule nothing.
  if not IsRunnableTask(t):
    return True

  # Check that all dependencies have already been scheduled:
  for dep_name in t.deps:
    if dep_name == t:
      raise SchedulingError("task '" + t + "' depends on itself")

    if dep_name not in schedule:
      # If any of the dependencies are not runnable, schedule nothing.
      if ScheduleTask(dep_name, schedule, all_tasks, depth + 1):
        return True

  # Schedule the task if necessary:
  if task_name not in schedule:
    schedule.append(task_name)


def GetTasksToRun(task_names):
  """ generate the list of task names to run """
  # Remove duplicate task names:
  task_names = set(task_names)

  with open(
      os.path.expanduser('~/phd/experimental/system/dotfiles/test.py')) as f:
    exec(f.read(), globals(), locals())

  # Determine the tasks which need scheduling:
  to_schedule = task_names if len(task_names) else TASKS.keys()
  # Build the schedule:
  to_schedule = collections.deque(sorted(to_schedule))
  schedule = []
  try:
    while len(to_schedule):
      task_name = to_schedule.popleft()
      ScheduleTask(task_name, schedule, TASKS.keys())
  except SchedulingError as e:
    logging.critical("fatal: " + str(e))
    sys.exit(1)

  return collections.deque([TASKS[t] for t in schedule])


def AssertSet(value, name):
  if value is None:
    raise MissingRequiredArgument(name)
  return value


class MissingRequiredArgument(TaskArgumentError):

  def __init__(self, argument):
    super(MissingRequiredArgument, self).__init__(argument, 'not set')
