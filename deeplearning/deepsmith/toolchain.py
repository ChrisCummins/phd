"""This file implements the programming language toolchain abstraction."""
from deeplearning.deepsmith import db


class Toolchain(db.StringTable):
  id_t = db.StringTable.id_t
  __tablename__ = "toolchains"
