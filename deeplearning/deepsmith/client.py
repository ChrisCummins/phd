"""This file implements the client class. A client is a physical machine."""
from deeplearning.deepsmith import db


class Client(db.StringTable):
  id_t = db.StringTable.id_t
  __tablename__ = "clients"
