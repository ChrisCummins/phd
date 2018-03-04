"""This file implements the programming language toolchain abstraction."""
from deeplearning.deepsmith import db


class Toolchain(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "toolchains"
