# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines telemetry data gathers."""
import pathlib
import re
import typing

import sqlalchemy as sql

from labm8 import app
from labm8 import jsonutil
from labm8 import labdate
from labm8 import pbutil
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = sqlutil.Base()


class ConfigProto(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):

  id: int = sql.Column(sql.Integer, primary_key=True)
  name: str = sql.Column(sql.String())
  proto = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)


class DashboardDatabase(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool):
    super(DashboardDatabase, self).__init__(url, Base, must_exist=must_exist)
