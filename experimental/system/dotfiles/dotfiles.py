"""This file implements the declarative functions for dotfiles."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time

from experimental.system.dotfiles.implementation import context
from experimental.system.dotfiles.implementation import host
from experimental.system.dotfiles.implementation import io
from experimental.system.dotfiles.implementation import task


def BuildArgumentParser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'tasks',
      metavar='<task>',
      nargs='*',
      help="the name of tasks to run (default: all)")
  action_group = parser.add_mutually_exclusive_group()
  action_group.add_argument('-d', '--describe', action="store_true")
  action_group.add_argument('-u', '--upgrade', action='store_true')
  action_group.add_argument('-r', '--remove', action='store_true')
  action_group.add_argument('--versions', action="store_true")
  verbosity_group = parser.add_mutually_exclusive_group()
  verbosity_group.add_argument('-v', '--verbose', action='store_true')
  return parser


def main(argv):
  # Parse arguments
  parser = BuildArgumentParser()
  args = parser.parse_args(argv)

  # Configure logger
  io.SetVerbosity(verbose=args.verbose)

  # Get the list of tasks to run
  app.Debug("creating tasks list ...")
  queue = task.GetTasksToRun(args.tasks)
  done = set()
  ntasks = len(queue)

  fmt_bld, fmt_end, fmt_red = io.Colors.BOLD, io.Colors.END, io.Colors.RED

  # --describe flag prints a description of the work to be done:
  platform = host.GetPlatform()
  if args.describe:
    msg = ("There are {fmt_bld}{ntasks}{fmt_end} tasks to run on {platform}:".
           format(**vars()))
    app.Info(msg)
    for i, t in enumerate(queue):
      task_name = type(t).__name__
      j = i + 1
      desc = type(t).__name__
      msg = (
          "[{j:2d}/{ntasks:2d}]  {fmt_bld}{task_name}{fmt_end} ({desc})".format(
              **vars()))
      app.Info(msg)
      # build a list of generated files
      for file in t.genfiles:
        app.Debug("    " + os.path.abspath(os.path.expanduser(file)))

    return 0

  # --versions flag prints the specific task versions:
  if args.versions:
    for i, t in enumerate(queue):
      for name in sorted(t.versions.keys()):
        task_name = type(t).__name__
        version = t.versions[name]
        app.Info("{task_name}:{name}=={version}".format(**vars()))
    return 0

  if args.upgrade:
    task_type = "upgrade"
  elif args.remove:
    task_type = "uninstall"
  else:
    task_type = "install"

  msg = ("Running {fmt_bld}{ntasks} {task_type}{fmt_end} tasks on {platform}:".
         format(**vars()))
  app.Info(msg)

  # Run the tasks
  ctx = context.CallContext()
  errored = False
  try:
    for i, t in enumerate(queue):
      task_name = type(t).__name__

      j = i + 1
      msg = "[{j:2d}/{ntasks:2d}] {fmt_bld}{task_name}{fmt_end} ...".format(
          **vars())
      app.Info(msg)

      start_time = time.time()

      # Resolve and run install() method:
      t(ctx)
      done.add(t)

      # Ensure that genfiles have been generated:
      if task_type == "install":
        for file in t.genfiles:
          file = os.path.abspath(os.path.expanduser(file))
          app.Debug("assert exists: '{file}'".format(**vars()))
          if not (os.path.exists(file) or host.CheckShellCommand(
              "sudo test -f '{file}'".format(**vars())) or
                  host.CheckShellCommand(
                      "sudo test -d '{file}'".format(**vars()))):
            raise task.InvalidTaskError(
                'genfile "{file}" not created'.format(**vars()))
      runtime = time.time() - start_time

      app.Debug("{task_name} task completed in {runtime:.3f}s".format(**vars()))
      sys.stdout.flush()
  except KeyboardInterrupt:
    app.Info("\ninterrupt")
    errored = True
  except Exception as e:
    e_name = type(e).__name__
    app.Error("{fmt_bld}{fmt_red}fatal error: {e_name}".format(**vars()))
    app.Error(str(e) + io.Colors.END)
    errored = True
    if logging.getLogger().level <= logging.DEBUG:
      raise
  finally:
    # Task teardowm
    app.Debug(io.Colors.BOLD + "Running teardowns" + io.Colors.END)
    for t in done:
      t.TearDown(ctx)

      # build a list of temporary files
      tmpfiles = getattr(t, "__tmpfiles__", [])
      tmpfiles += getattr(t, "__" + host.GetPlatform() + "_tmpfiles__", [])

      # remove any temporary files
      for file in tmpfiles:
        file = os.path.abspath(os.path.expanduser(file))
        if os.path.exists(file):
          app.Debug("rm {file}".format(**vars()))
          os.remove(file)

  return 1 if errored else 0


if __name__ == '__main__':
  main(sys.argv[1:])
