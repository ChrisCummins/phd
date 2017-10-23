#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Database backend.

Some notes on schema portability:
    * SQLite auto incrementing requires that integral indices be integers.
      As a result, we use the with_variant() method to case non-integer id_t
      to integers on sqlite. See:
        http://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#sqlite-auto-incrementing-behavior
"""
import datetime
import humanize
import logging
import os
import progressbar
import re
import sqlalchemy as sql
import threading

from contextlib import contextmanager
from labm8 import crypto, fs, prof, system
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.declarative import declarative_base
from signal import Signals
from sqlalchemy.sql import func
from time import time
from typing import Dict, Iterable, List, Tuple, Union

import dsmith
from dsmith import Colors


session_t = sql.orm.session.Session
query_t = sql.orm.query.Query


def get_or_add(session: sql.orm.session.Session, model,
               defaults: Dict[str, object]=None, **kwargs) -> object:
    """
    Instantiate a mapped database object. If the object is not in the database,
    add it.

    Note that no change is written to disk until commit() is called on the
    session.
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if not instance:
        params = dict((k, v) for k, v in kwargs.items()
                      if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)

        # logging
        logging.debug(f"new {model.__name__} record")

    return instance


def paginate(query: query_t, page_size: int=1000):
    """
    Paginate query results.
    """
    offset = 0
    while True:
        r = False
        for elem in query.limit(page_size).offset(offset):
           r = True
           yield elem
        offset += page_size
        if not r:
            break


class Proxy(object):
    """
    A proxy object is used to store all of the information required to create a
    database record, without needing to be bound to the lifetime of a database
    session.
    """
    def to_record(self, session: session_t) -> 'Base':
        """
        Instantiate a database record from this proxy.
        """
        raise NotImplementedError("abstract class")


def save_proxies(session: session_t, proxies: List[Proxy],
                 max_attempts: int=3, attempt: int=1,
                 exception=None) -> None:
    """
    Convert a set of proxy objects in to database records and save them.

    Raises:
        OSError: In case of error importing proxies.
    """
    # There is a potential race condition when multiple
    # harnesses are adding database records with unique key
    # constraints. Rather than figure out a proper
    # serialization strategy, I find it's easier just to
    # retry a few times. I'm a terrible person.
    max_attempts = 3

    # break early if there's nothing to import
    nproxies = len(proxies)
    if not nproxies:
        return

    if attempt > max_attempts:
        # Fallback to sequential import of all proxies. Ignore all errors, this
        # is just trying to minimize information lost. Try and get as many
        # proxies in the database as possible before error-ing:
        for proxy in proxies:
            try:
                session.add(proxy.to_record(session))
                session.commit()
            except:
                pass

        msg = f"{max_attempts} consecutive database errors, aborting"
        if exception:
            raise OSError(msg) from exception
        else:
            raise OSError(msg)

    logging.debug(f"flushing {nproxies} records")
    try:
        start_time = time()
        session.add_all(proxy.to_record(session) for proxy in proxies)
        session.commit()
        runtime = time() - start_time
        logging.info(f"flushed {nproxies} records in {runtime:.2} seconds")
    except IntegrityError as e:
        logging.debug(e)
        logging.warning("database integrity error, rolling back")
        session.rollback()
        save_proxies(session, proxies, max_attempts, attempt + 1, e)
    except OperationalError as e:
        logging.debug(e)
        logging.warning("database operational error, rolling back")
        session.rollback()
        save_proxies(session, proxies, max_attempts, attempt + 1, e)
