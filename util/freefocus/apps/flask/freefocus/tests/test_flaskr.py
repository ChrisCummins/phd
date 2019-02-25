# -*- coding: utf-8 -*-
"""
    Flaskr Tests
    ~~~~~~~~~~~~

    Tests the Flaskr application.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

import os
import tempfile

import pytest
from freefocus import freefocus


@pytest.fixture
def client(request):
  db_fd, freefocus.app.config['DATABASE'] = tempfile.mkstemp()
  freefocus.app.config['TESTING'] = True
  client = freefocus.app.test_client()
  with freefocus.app.app_context():
    freefocus.init_db()

  def teardown():
    os.close(db_fd)
    os.unlink(freefocus.app.config['DATABASE'])

  request.addfinalizer(teardown)

  return client


def login(client, username, password):
  return client.post(
      '/login',
      data=dict(username=username, password=password),
      follow_redirects=True)


def logout(client):
  return client.get('/logout', follow_redirects=True)


def test_empty_db(client):
  """Start with a blank database."""
  rv = client.get('/')
  assert b'No entries here so far' in rv.data


def test_login_logout(client):
  """Make sure login and logout works"""
  rv = login(client, freefocus.app.config['USERNAME'],
             freefocus.app.config['PASSWORD'])
  assert b'You were logged in' in rv.data
  rv = logout(client)
  assert b'You were logged out' in rv.data
  rv = login(client, freefocus.app.config['USERNAME'] + 'x',
             freefocus.app.config['PASSWORD'])
  assert b'Invalid username' in rv.data
  rv = login(client, freefocus.app.config['USERNAME'],
             freefocus.app.config['PASSWORD'] + 'x')
  assert b'Invalid password' in rv.data


def test_messages(client):
  """Test that messages work"""
  login(client, freefocus.app.config['USERNAME'],
        freefocus.app.config['PASSWORD'])
  rv = client.post(
      '/add',
      data=dict(title='<Hello>', text='<strong>HTML</strong> allowed here'),
      follow_redirects=True)
  assert b'No entries here so far' not in rv.data
  assert b'&lt;Hello&gt;' in rv.data
  assert b'<strong>HTML</strong> allowed here' in rv.data
