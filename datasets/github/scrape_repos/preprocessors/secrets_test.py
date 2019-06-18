"""Unit tests for //TODO:datasets.github.scrape_repos.preprocessors/secrets_test."""

import pytest
from datasets.github.scrape_repos.preprocessors import secrets
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_TODO():
  """Test a text that doesn't contain a secret."""
  assert secrets.ScanForSecrets('Hello, world!')


def test_ScanForSecrets_rsa_private_key():
  """Test a (false positive) text that may contain a secret."""
  with pytest.raises(secrets.TextContainsSecret) as e_ctx:
    secrets.ScanForSecrets("""
-----BEGIN RSA PRIVATE KEY-----
""")
  assert str(e_ctx.value) == 'PrivateKeyDetector'


if __name__ == '__main__':
  test.Main()
