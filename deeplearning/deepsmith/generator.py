"""This file defines the testcase generator."""
from datetime import datetime
from typing import List

from sqlalchemy import Integer, Column, DateTime, String, UniqueConstraint
from sqlalchemy.orm import relationship

from deeplearning.deepsmith import db


class Generator(db.Base):
  id_t = Integer
  __tablename__ = "generators"

  # Columns:
  id: int = Column(id_t, primary_key=True)
  date_added: datetime = Column(DateTime, nullable=False, default=db.now)
  name: str = Column(String(1024), nullable=False)
  version: str = Column(String(1024), nullable=False)

  # Relationships:
  testcases: List["Testcase"] = relationship("Testcase", back_populates="generator")

  # Constraints:
  __table_args__ = (
    UniqueConstraint('name', 'version', name='unique_generator'),
  )
