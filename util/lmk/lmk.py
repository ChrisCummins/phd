#!/usr/bin/env python
# Copyright 2017-2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""let me know - Email output of command upon completion

Attributes
----------
__description__ : str
    Package description.

DEFAULT_CFG : str
    Default configuration.

DEFAULT_CFG_PATH : str
    Default path to the configuration file.

E_CFG : int
    Non-zero return code for configuration errors.

E_SMTP : int
    Non-zero return code for fatal SMTP errors.
"""
from __future__ import print_function

import argparse
import cgi
import os
import smtplib
import socket
import string
import subprocess
import sys
from datetime import datetime
from datetime import timedelta
from email.mime.application import MIMEApplication


# Python 2 and 3 have different email module layouts:
if sys.version_info >= (3, 0):
  from email.mime.multipart import MIMEMultipart
  from email.mime.text import MIMEText
else:
  from email.MIMEMultipart import MIMEMultipart
  from email.MIMEText import MIMEText

DEFAULT_CFG_PATH = os.path.expanduser('~/.lmkrc')

__description__ = """\
{bin}: let me know. Patiently awaits the completion of the
specified command, and emails you with the output and result.

Examples
--------
Run a command using lmk to receive an email when it completes, containing its
output and return code:

    $ lmk './experiments -n 100'

Alternatively, pipe the output of commands to lmk to receive an email when they
complete:

    $ (./experiment1.sh; experiment2.py -n 100) 2>&1 | lmk -

Configuration
-------------
The file {cfg} contains the configuration settings. Modify
the smtp and message settings to suit.

Made with \033[1;31m♥\033[0;0m by Chris Cummins.
<https://github.com/ChrisCummins/phd>\
""".format(
    bin=sys.argv[0], cfg=DEFAULT_CFG_PATH)

DEFAULT_CFG = """\
; lkm config <https://github.com/ChrisCummins/phd>
; Configure smtp section to your outgoing mailbox.
; Shell variables are expanded in this file.

[smtp]
Host: smtp.gmail.com
Port: 587
Username: $LMK_USER
Password: $LMK_PWD

[exec]
Shell: /bin/bash

[messages]
From: $USER@$HOST
To: $MAILTO
"""

E_CFG = 2
E_SMTP = 3


class colors:
  """
  Shell escape codes.
  """
  reset = '\033[0;0m'
  red = '\033[1;31m'
  blue = '\033[1;34m'
  cyan = '\033[1;36m'
  green = '\033[0;32m'
  bold = '\033[;1m'
  reverse = '\033[;7m'


class ArgumentParser(argparse.ArgumentParser):
  """
  Specialized argument parser, with --version flag.
  """

  def __init__(self, *args, **kwargs):
    """
    See python argparse.ArgumentParser.__init__().
    """
    super(ArgumentParser, self).__init__(*args, **kwargs)
    self.add_argument(
        '--version',
        action='store_true',
        help='show version information and exit')
    self.add_argument(
        '--create-config',
        action='store_true',
        help='create configuration file and exit')

  def parse_args(self, args=sys.argv[1:], namespace=None):
    """
    See python argparse.ArgumentParser.parse_args().
    """
    # --version option overrides the normal argument parsing process.
    if '--version' in args:
      print('lmk master, made with {c.red}♥{c.reset} by '
            'Chris Cummins <chrisc.101@gmail.com>'.format(c=colors))
      sys.exit(0)

    if '--create-config' in args:
      get_cfg_path()
      sys.exit(0)

    return super(ArgumentParser, self).parse_args(args, namespace)


#### From humanize.
# To remove the dependency on any pip package and make this script
# self-contained, I have inlined the naturaltime() function from Jason's
# excellent humanize library. See: <https://github.com/jmoiron/humanize>.
#
# Copyright (c) 2010 Jason Moiron and Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def abs_timedelta(delta):
  """Returns an "absolute" value for a timedelta, always representing a
  time distance."""
  if delta.days < 0:
    now = datetime.now()
    return now - (now + delta)
  return delta


def date_and_delta(value):
  """Turn a value into a date and a timedelta which represents how long ago
  it was.  If that's not possible, return (None, value)."""
  now = datetime.now()
  if isinstance(value, datetime):
    date = value
    delta = now - value
  elif isinstance(value, timedelta):
    date = now - value
    delta = value
  else:
    try:
      value = int(value)
      delta = timedelta(seconds=value)
      date = now - delta
    except (ValueError, TypeError):
      return (None, value)
  return date, abs_timedelta(delta)


def _(message):
  return message


def ngettext(message, plural, num):
  if num == 1:
    return message
  else:
    return plural


def naturaldelta(value, months=True):
  """Given a timedelta or a number of seconds, return a natural
  representation of the amount of time elapsed.  This is similar to
  ``naturaltime``, but does not add tense to the result.  If ``months``
  is True, then a number of months (based on 30.5 days) will be used
  for fuzziness between years."""
  date, delta = date_and_delta(value)
  if date is None:
    return value

  use_months = months

  seconds = abs(delta.seconds)
  days = abs(delta.days)
  years = days // 365
  days = days % 365
  months = int(days // 30.5)

  if not years and days < 1:
    if seconds == 0:
      return _("a moment")
    elif seconds == 1:
      return _("a second")
    elif seconds < 60:
      return ngettext("%d second", "%d seconds", seconds) % seconds
    elif 60 <= seconds < 120:
      return _("a minute")
    elif 120 <= seconds < 3600:
      minutes = seconds // 60
      return ngettext("%d minute", "%d minutes", minutes) % minutes
    elif 3600 <= seconds < 3600 * 2:
      return _("an hour")
    elif 3600 < seconds:
      hours = seconds // 3600
      return ngettext("%d hour", "%d hours", hours) % hours
  elif years == 0:
    if days == 1:
      return _("a day")
    if not use_months:
      return ngettext("%d day", "%d days", days) % days
    else:
      if not months:
        return ngettext("%d day", "%d days", days) % days
      elif months == 1:
        return _("a month")
      else:
        return ngettext("%d month", "%d months", months) % months
  elif years == 1:
    if not months and not days:
      return _("a year")
    elif not months:
      return ngettext("1 year, %d day", "1 year, %d days", days) % days
    elif use_months:
      if months == 1:
        return _("1 year, 1 month")
      else:
        return ngettext("1 year, %d month", "1 year, %d months",
                        months) % months
    else:
      return ngettext("1 year, %d day", "1 year, %d days", days) % days
  else:
    return ngettext("%d year", "%d years", years) % years


def naturaltime(value, future=False, months=True):
  """Given a datetime or a number of seconds, return a natural representation
  of that time in a resolution that makes sense.  This is more or less
  compatible with Django's ``naturaltime`` filter.  ``future`` is ignored for
  datetimes, where the tense is always figured out based on the current time.
  If an integer is passed, the return value will be past tense by default,
  unless ``future`` is set to True."""
  now = datetime.now()
  date, delta = date_and_delta(value)
  if date is None:
    return value
  # determine tense by value only if datetime/timedelta were passed
  if isinstance(value, (datetime, timedelta)):
    future = date > now

  ago = _('%s from now') if future else _('%s ago')
  delta = naturaldelta(delta, months)

  if delta == _("a moment"):
    return _("now")

  return ago % delta


#### End humanize.


def parse_args(args):
  """
  Parse command line options.

  Returns
  -------
  str
      Command to execute.
  """
  parser = ArgumentParser(
      description=__description__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      '-e',
      '--only-errors',
      action='store_true',
      help='only notify if command fails')
  parser.add_argument(
      'command',
      metavar='<command>',
      help='command to execute, or "-" to read from stdin')
  return parser.parse_args(args)


def create_default_cfg(path):
  """
  Create default configuration file.

  Parameters
  ----------
  path : str
      Path of cfg file to create.
  """
  with open(path, 'w') as outfile:
    print(DEFAULT_CFG, end='', file=outfile)
  os.chmod(path, 384)  # 384 == 0o600
  print(
      '{c.bold}[lmk] created default configuration file {path}{c.reset}'.format(
          c=colors, path=path),
      file=sys.stderr)


def parse_str(str_, substitutions={}):
  """
  Parse a string, escaping shell and special variables.

  Rudimentary, crummy bash variable parser.

  Parameters
  ----------
  str_ : str
      String to parse.
  substitutions : Dict[str, lambda: str]
      A dictionary of substitution functions.
  """

  def expandvar():
    if ''.join(varname) in substitutions:
      var = substitutions[''.join(varname)]()
    else:
      var = os.environ.get(''.join(varname), '')
    out.append(var)

  BASH_VAR_CHARS = string.ascii_letters + string.digits + '_'

  # parser state
  out = []
  varname = []
  invar = False
  escape = False

  for c in str_:
    if c == '\\':
      if escape:
        # '\\' -> '\'
        out.append('\\')
        escape = False
      else:
        escape = True
    elif c == '$':
      if escape:
        # '\$' -> '$'
        out.append('$')
        escape = False
      else:
        if invar:
          # '$foo$bar' -> $(foo) $(bar)
          expandvar()
        varname = []
        invar = True
    elif c == ' ':
      escape = False
      if invar:
        # '$foo ' -> $(foo)' '
        expandvar()
        varname = []
        invar = False
      out.append(' ')
    else:
      if invar:
        if c in BASH_VAR_CHARS:
          varname.append(c)
        else:
          # '$foo@' -> $(foo)'@'
          expandvar()
          varname = []
          invar = False
          out.append(c)
      else:
        escape = False
        out.append(c)

  if invar:
    expandvar()
  return ''.join(out)


def load_cfg(path=None):
  """
  Parse configuration.

  In case of error, kills process with status E_CFG.

  Returns
  -------
  ConfigParser
      Parsed configuration.
  """

  def _verify(stmt, *msg, **kwargs):
    sep = kwargs.get('sep', ' ')
    if not stmt:
      print(
          '{c.bold}{c.red}[lmk] {msg}{c.reset}'.format(
              c=colors, msg=sep.join(msg)),
          file=sys.stderr)
      sys.exit(E_CFG)

  if sys.version_info >= (3, 0):
    from configparser import ConfigParser
  else:
    from ConfigParser import ConfigParser

  if path is None:
    path = get_cfg_path()

  cfg = ConfigParser()
  cfg.read(path)

  _verify('smtp' in cfg, 'config file %s contains no [smtp] section' % path)
  _verify('host' in cfg['smtp'], 'no host in %s:smtp' % path)
  _verify('port' in cfg['smtp'], 'no port in %s:smtp' % path)
  _verify('username' in cfg['smtp'], 'no username in %s:smtp' % path)
  _verify('password' in cfg['smtp'], 'no password in %s:smtp' % path)

  _verify('messages' in cfg,
          'config file %s contains no [messages] section' % path)
  _verify('from' in cfg['messages'], 'no from address in %s:messages' % path)
  _verify('to' in cfg['messages'], 'no to address in %s:messages' % path)

  parse = lambda x: parse_str(x, {'HOST': lambda: socket.gethostname()})

  cfg['smtp']['host'] = parse(cfg['smtp']['host'])
  cfg['smtp']['port'] = parse(cfg['smtp']['port'])
  cfg['smtp']['username'] = parse(cfg['smtp']['username'])
  cfg['smtp']['password'] = parse(cfg['smtp']['password'])

  _verify(cfg['smtp']['host'], 'stmp host is empty. Check %s' % path)
  _verify(cfg['smtp']['port'], 'stmp port is empty. Check %s' % path)
  _verify(cfg['smtp']['username'], 'stmp username is empty. Check %s' % path)
  _verify(cfg['smtp']['password'], 'stmp password is empty. Check %s' % path)

  cfg['messages']['from'] = parse(cfg['messages']['from'])
  cfg['messages']['to'] = parse(cfg['messages']['to'])
  # note: 'subject' variables are parsed after command completion,
  #   so we can substitue in outcomes.

  if 'exec' not in cfg:
    cfg.add_section('exec')
  if 'shell' not in cfg['exec']:
    cfg['exec']['shell'] = '/bin/sh'

  # add runtime metadata
  cfg.add_section('/run')
  cfg['/run']['path'] = path

  return cfg


def get_smtp_server(cfg):
  """
  Create a connection an SMTP server.

  In case of an error, this function kills the process.
  Remove to close connections with quit().

  Parameters
  ----------
  cfg : ConfigParser
      Configuration.

  Returns
  -------
  SMTP
      SMTP Server.
  """

  def _error(*msg, **kwargs):
    sep = kwargs.get('sep', ' ')
    print(
        '{c.bold}{c.red}[lmk] {msg}{c.reset}'.format(
            c=colors, msg=sep.join(msg)),
        file=sys.stderr)
    sys.exit(E_SMTP)

  try:
    server = smtplib.SMTP(cfg['smtp']['host'], int(cfg['smtp']['port']))
    server.starttls()
    server.login(cfg['smtp']['username'], cfg['smtp']['password'])
    return server
  except smtplib.SMTPHeloError:
    _error('connection to {host}:{port} failed'.format(
        host=cfg['smtp']['host'], port=cfg['smtp']['port']))
  except smtplib.SMTPAuthenticationError:
    _error('smtp authentication failed. Check username and password in '
           '%s' % cfg['/run']['path'])
  except smtplib.SMTPServerDisconnected:
    _error(
        '{host}:{port} disconnected. Check smtp settings in {cfg_path}'.format(
            host=cfg['smtp']['host'],
            port=cfg['smtp']['port'],
            cfg_path=cfg['/run']['path']),
        file=sys.stderr)
  except smtplib.SMTPException:
    _error('unknown error from {host}:{port}'.format(
        host=cfg['smtp']['host'], port=cfg['smtp']['port']))


def send_email_smtp(cfg, server, msg):
  """
  Send an email.

  Parameters
  ----------
  server : SMTP
      SMTP server.
  msg : MIMEMultipart
      Message to send.

  Returns
  -------
  bool
      True is send suceeded, else false.
  """

  def _error(*msg, **kwargs):
    sep = kwargs.get('sep', ' ')
    print(
        '{c.bold}{c.red}[lmk] {msg}{c.reset}'.format(
            c=colors, msg=sep.join(msg)),
        file=sys.stderr)
    return False

  recipient = msg['To'].strip()
  if not recipient:
    return _error('no recipient')

  try:
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    print(
        '{c.bold}{c.cyan}[lmk] {recipient} notified{c.reset}'.format(
            c=colors, recipient=recipient),
        file=sys.stderr)
    return True
  except smtplib.SMTPHeloError:
    return _error('connection to {host}:{port} failed'.format(
        host=cfg['smtp']['host'], port=cfg['smtp']['port']))
  except smtplib.SMTPDataError:
    return _error('unknown error from {host}:{port}'.format(
        host=cfg['smtp']['host'], port=cfg['smtp']['port']))
  except smtplib.SMTPRecipientsRefused:
    return _error('recipient {recipient} refused'.format(recipient=recipient))
  except smtplib.SMTPSenderRefused:
    return _error('sender {from_} refused'.format(from_=msg['From']))
  return False


def build_html_message_body(output,
                            command=None,
                            returncode=None,
                            date_started=None,
                            date_ended=None,
                            runtime=None,
                            snip_after=220,
                            snip_to=200):
  """
  Parameters
  ----------
  command : str
      The command which was run.
  output : str
      The output of the command.
  runtime : Tuple(datetime, datetime)
      The command start and end dates.
  snip_after : int (optional)
      The maximum number of lines to permit before snipping message.
  snip_to : int optional
      If the number of lines exceeds snip_after, snip to this many number of
      lines.

  Returns
  -------
  Tuple(str, bool)
      The HTML body string, and a boolean value signifying whether the
      output was tuncated.
  """
  if snip_to > snip_after:
    raise ValueError("snip_to must be <= snip_after")

  user = os.environ['USER']
  host = socket.gethostname()
  cwd = os.getcwd()
  lmk = '<a href="github.com/ChrisCummins/phd">lmk</a>'
  me = '<a href="http://chriscummins.cc">Chris Cummins</a>'

  prompt_css = ";".join([
      "font-family:'Courier New', monospace",
      "font-weight:700",
      "font-size:14px",
      "padding-right:10px",
      "color:#000",
      "text-align:right",
  ])

  command_css = ";".join([
      "font-family:'Courier New', monospace",
      "font-weight:700",
      "font-size:14px",
      "color:#000",
  ])

  lineno_css = ";".join([
      "font-family:'Courier New', monospace",
      "font-size:14px",
      "padding-right:10px",
      "color:#666",
      "text-align:right",
  ])

  line_css = ";".join([
      "font-family:'Courier New', monospace",
      "font-size:14px",
      "color:#000",
  ])

  # metadata block
  html = '<table>\n'
  style = 'padding-right:15px;'
  if date_started:
    delta = naturaltime(datetime.now() - date_started)
    html += (u'  <tr><td style="{style}">Started</td>'
             u'<td>{date_started} ({delta})</td></tr>\n'.format(
                 style=style, date_started=date_started, delta=delta))
  if date_ended:
    html += (u'  <tr><td style="{style}">Completed</td>'
             u'<td>{date_ended}</td></tr>\n'.format(
                 style=style, date_ended=date_ended))
  if returncode is not None:
    html += (u'  <tr><td style="{style}">Return code</td>'
             u'<td style="font-weight:700;">{returncode}</td></tr>\n'.format(
                 style=style, returncode=returncode))
  html += (u'  <tr><td style="{style}">Working directory</td>'
           u'<td>{cwd}</td></tr>\n'.format(style=style, cwd=cwd))
  html += '</table>\n<hr style="margin-top:20px;"/>\n'

  # output
  html += '<table>\n'

  # command line invocation
  if command is not None:
    command_html = cgi.escape(command)
    html += u"""\
  <tr style="line-height:1em;">
    <td style="{prompt_css}">$</td>
    <td style="{command_css}">{command_html}</td>
  </tr>
""".format(
        prompt_css=prompt_css,
        command_css=command_css,
        command_html=command_html)

  # command output
  lines = output.split('\n')
  truncated = False

  if len(lines) > snip_after:
    truncated = True
    # truncated report. First and last lines of output
    line_nums = range(1, snip_to // 2 + 1)
    for line, lineno in zip(lines[:snip_to // 2], line_nums):
      line_html = cgi.escape(line)
      html += u"""\
  <tr style="line-height:1em;">
    <td style="{lineno_css}">{lineno}</td>
    <td style="{line_css}">{line_html}</td>
  </tr>
""".format(
          lineno_css=lineno_css,
          lineno=lineno,
          line_css=line_css,
          line_html=line_html)
    num_omitted = len(lines) - 200
    html += "</table>"
    html += "... ({num_omitted} lines snipped)".format(num_omitted=num_omitted)
    html += "<table>\n"
    line_nums = range(len(lines) - snip_to // 2 + 1, len(lines) + 1)
    for line, lineno in zip(lines[-snip_to // 2:], line_nums):
      line_html = cgi.escape(line)
      html += u"""\
  <tr style="line-height:1em;">
    <td style="{lineno_css}">{lineno}</td>
    <td style="{line_css}">{line_html}</td>
  </tr>
""".format(
          lineno_css=lineno_css,
          lineno=lineno,
          line_css=line_css,
          line_html=line_html)
  else:
    # full length report
    for line, lineno in zip(lines, range(1, len(lines) + 1)):
      try:
        line = line.decode('utf-8')
      except AttributeError:  # str.decode() depends on Python version.
        pass
      line_html = cgi.escape(line)
      html += u"""
  <tr style="line-height:1em;">
    <td style="{lineno_css}">{lineno}</td>
    <td style="{line_css}">{line_html}</td>
  </tr>
""".format(
          lineno_css=lineno_css,
          lineno=lineno,
          line_css=line_css,
          line_html=line_html)

  html += u'</table>\n'

  # footer
  html += u"""\
</table>

<hr style="margin-top:20px;"/>
<center style="color:#626262;">
  {lmk} made with ♥ by {me}
</center>
""".format(
      lmk=lmk, me=me)

  return html, truncated


def get_cfg_path():
  """
  Get path to config file.

  If config file not found, kills the process with E_CFG.

  Returns
  -------
  str
      Config path.
  """
  cfg_path = os.path.expanduser(os.environ.get('LMK_CFG', DEFAULT_CFG_PATH))
  if not os.path.exists(cfg_path) and cfg_path == DEFAULT_CFG_PATH:
    create_default_cfg(cfg_path)
  elif not os.path.exists(cfg_path):
    print(
        '{c.bold}{c.red}$LMK_CFG ({cfg_path}) not found{c.reset}'.format(
            c=colors, cfg_path=cfg_path),
        file=sys.stderr)
    sys.exit(E_CFG)
  return cfg_path


def check_connection(cfg=None):
  if cfg is None:
    cfg = load_cfg()

  get_smtp_server(cfg).quit()


def build_message_subject(output,
                          command=None,
                          returncode=None,
                          cfg=None,
                          date_started=None,
                          date_ended=None):
  """
  Build message subject line.

  Returns
  -------
  str
      Unicode message subject.
  """
  user = os.environ['USER']
  host = socket.gethostname()

  if command is not None and returncode is not None:
    happy_sad = u'🙈' if returncode else u'✔'
    return u'{user}@{host} {happy_sad} $ {command}'.format(
        user=user, host=host, happy_sad=happy_sad, command=command)
  elif command is not None:
    return u'{user}@{host} $ {command}'.format(
        user=user, host=host, command=command)
  elif date_started is not None:
    delta = naturaltime(datetime.now() - date_started)
    return u'{user}@{host} finished job started {delta}'.format(
        user=user, host=host, delta=delta)
  else:
    return u'{user}@{host} finished job'.format(user=user, host=host)


def let_me_know(output,
                command=None,
                returncode=None,
                cfg=None,
                date_started=None,
                date_ended=None):
  """Let me know: send an email to user.

  Args:
    output: The output of the command.
    command: The command that was executed.
    returncode: The returncode of the command.
    cfg: The configuration to use. If not provided, the default configuration
      is used.
    date_started: A datetime object representing the start of execution.
    date_ended: A datetime object representing the end of execution.
  """
  if cfg is None:
    cfg = load_cfg()

  subject = build_message_subject(
      output=output,
      command=command,
      returncode=returncode,
      date_started=date_started,
      date_ended=date_ended)
  html, truncated = build_html_message_body(
      output=output,
      command=command,
      returncode=returncode,
      date_started=date_started,
      date_ended=date_ended)
  if sys.version_info < (3, 0):
    html = html.encode('utf-8')

  msg = MIMEMultipart()
  msg['From'] = cfg['messages']['from']
  msg['Subject'] = subject
  msg.attach(MIMEText(html, 'html'))

  if truncated:
    attachment = MIMEApplication(output, Name="output.txt")
    attachment['Content-Disposition'] = 'attachment; filename="output.txt"'
    msg.attach(attachment)

  server = get_smtp_server(cfg)
  for recipient in cfg['messages']['to'].split(','):
    msg['To'] = recipient
    send_email_smtp(cfg, server, msg)
  server.quit()


def read_from_stdin():
  cfg = load_cfg()
  check_connection(cfg)

  date_started = datetime.now()

  out = []
  for line in sys.stdin:
    sys.stdout.write(line)
    out.append(line)

  date_ended = datetime.now()
  output = ''.join(out).rstrip()

  let_me_know(
      output=output, cfg=cfg, date_started=date_started, date_ended=date_ended)


def run_subprocess(command, only_errors=False):
  cfg = load_cfg()
  check_connection(cfg)

  date_started = datetime.now()

  out = []
  process = subprocess.Popen(
      command,
      shell=True,
      executable=cfg['exec']['shell'],
      universal_newlines=True,
      bufsize=1,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT)

  if sys.version_info >= (3, 0):
    output_iter = process.stdout
  else:
    output_iter = iter(process.stdout.readline, b'')

  with process.stdout:
    for line in output_iter:
      sys.stdout.write(line)
      out.append(line)
  process.wait()

  date_ended = datetime.now()

  output = ''.join(out).rstrip()
  returncode = process.returncode

  if returncode or not only_errors:
    let_me_know(
        output=output,
        command=command,
        returncode=returncode,
        cfg=cfg,
        date_started=date_started,
        date_ended=date_ended)

  return returncode


def main():
  args = parse_args(sys.argv[1:])

  try:
    if args.command == '-':
      # check that command line usage is correct
      if args.only_errors:
        print('{c.bold}{c.red}[lmk] --only-errors option cannot be '
              'used with stdin{c.reset}'.format(c=colors))
        sys.exit(1)

      read_from_stdin()
    else:
      sys.exit(run_subprocess(args.command, only_errors=args.only_errors))
  except KeyboardInterrupt:
    print('{c.bold}{c.red}[lmk] aborted{c.reset}'.format(c=colors))
    sys.exit(1)


if __name__ == '__main__':
  main()
