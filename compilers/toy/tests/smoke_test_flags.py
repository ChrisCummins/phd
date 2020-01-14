"""This file defines the common flags for smoke tests."""
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_integer("stage", 1, "The stage to test.")
app.DEFINE_boolean("xfail", False, "Strictly require the test to fail.")
