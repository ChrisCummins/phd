"""This file implements the programming language abstraction."""
from deeplearning.deepsmith import db


class Language(db.ListOfNames):
  id_t = db.ListOfNames.id_t
  __tablename__ = "languages"
