"""
Database utility code.

Some notes on schema portability:
    * SQLite auto incrementing requires that integral indices be integers.
      As a result, we use the with_variant() method to case non-integer id_t
      to integers on sqlite. See:
        http://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#sqlite-auto-incrementing-behavior
"""
import typing
import os
import sqlalchemy as sql

from absl import logging

from labm8 import crypto, fs, prof, system
from sqlalchemy.exc import IntegrityError, OperationalError
from time import time

from deeplearning.deepsmith import db


def make_engine(**kwargs) -> sql.engine.Engine:
    """
    Raises:
        ValueError: If DB_ENGINE config value is invalid.
    """
    prefix = kwargs.get("prefix", "")
    engine = kwargs.get("engine", dsmith.DB_ENGINE)

    name = f"{prefix}dsmith_{dsmith.version_info.major}{dsmith.version_info.minor}"

    if engine == "mysql":
        username, password = kwargs.get("credentials", dsmith.DB_CREDENTIALS)
        hostname = kwargs.get("hostname", dsmith.DB_HOSTNAME)
        port = str(kwargs.get("port", dsmith.DB_PORT))

        # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
        # See: https://stackoverflow.com/a/16404147/1318051
        public_uri = f"mysql://{username}@{hostname}:{port}/{name}?charset=utf8".format(**vars())
        uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{name}?charset=utf8"
    elif engine == "sqlite":
        db_dir = kwargs.get("db_dir", dsmith.DB_DIR)
        if not db_dir:
            raise ValueError(f"no database directory specified")
        fs.mkdir(db_dir)  # create directory if it doesn't already exist
        path = fs.path(db_dir, f"{name}.db")
        uri = f"sqlite:///{path}"
        public_uri = uri
    else:
        raise ValueError(f"unsupported database engine {engine}")

    # Determine whether to enable logging of SQL statements:
    print("echo", os.environ.get("DB_DEBUG", False))
    echo = True if os.environ.get("DB_DEBUG", None) else False

    logging.debug("connecting to database %s", public_uri})
    return sql.create_engine(uri, encoding="utf-8", echo=echo), public_uri


def get_or_add(session: sql.orm.session.Session, model,
               defaults: typing.Dict[str, object]=None, **kwargs) -> object:
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
        logging.debug("new %s record", model.__name__)

    return instance


def paginate(query: db.query_t, page_size: int=1000):
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
    def to_record(self, session: db.session_t) -> 'Base':
        """
        Instantiate a database record from this proxy.
        """
        raise NotImplementedError("abstract class")


def save_proxies(session: db.session_t, proxies: typing.List[Proxy],
                 max_attempts: int=3) -> None:
    """
    Convert a set of proxy objects in to database records and save them.

    Raises:
        OSError: In case of error importing proxies.
    """
    return save_records(
        session, [proxy.to_record(session) for proxy in proxies], max_attempts)


def save_proxies_uniq_on(session: db.session_t, proxies: typing.List[Proxy], uniq_on: str,
                         max_attempts: int=3) -> int:
    return save_records_uniq_on(
        session, [proxy.to_record(session) for proxy in proxies], uniq_on, max_attempts)


def save_records_uniq_on(session: db.session_t, records: typing.List["Base"], uniq_on: str,
                         max_attempts: int=3) -> int:
    """ Save records which are unique on some column value. """
    # Break early if possible
    if not len(records):
        return

    # Filter duplicates in the list of new records:
    records = list(dict((getattr(record, uniq_on), record) for record in records).values())

    # Fetch a list of dupe keys already in the database:
    keys = [getattr(record, uniq_on) for record in records]
    table_col = getattr(type(records[0]), uniq_on)
    dupes = set(x[0] for x in session.query(table_col).filter(table_col.in_(keys)))

    # Filter the list of records to import, excluding dupes:
    uniq = [record for record in records if getattr(record, uniq_on) not in dupes]

    # Import those suckas:
    nprog, nuniq = len(records), len(uniq)
    save_records(session, uniq, max_attempts)

    logging.info("imported %s of %s unique programs", nuniq, nprog)
    return nuniq


def save_records(session: db.session_t, records: typing.List['Base'],
                 max_attempts: int=3, attempt: int=1,
                 exception=None) -> None:
    """
    Save a list of database records.

    Raises:
        OSError: In case of error importing proxies.
    """
    # break early if there's nothing to import
    nrecords = len(records)
    if not nrecords:
        return

    if attempt > max_attempts:
        # Fallback to sequential import of all records. Ignore all errors, this
        # is just trying to minimize information lost. Try and get as many
        # records in the database as possible before error-ing:
        for record in records:
            try:
                session.add(record)
                session.commit()
            except:
                pass

        msg = f"{max_attempts} consecutive database errors, aborting"
        if exception:
            raise OSError(msg) from exception
        else:
            raise OSError(msg)

    logging.debug("flushing %d records", nrecords)
    try:
        start_time = time()
        session.add_all(records)
        session.commit()
        runtime = time() - start_time
        logging.info("flushed %s records in %.2f seconds", nrecords, runtime)
    except IntegrityError as e:
        logging.debug(e)
        logging.warning("database integrity error, rolling back")
        logging.warning(e)
        session.rollback()
        save_records(session, records, max_attempts, attempt + 1, e)
    except OperationalError as e:
        logging.debug(e)
        logging.warning("database operational error, rolling back")
        logging.warning(e)
        session.rollback()
        save_records(session, records, max_attempts, attempt + 1, e)


def sql_query(*query):
    """ flatten an SQL query into a single line of text """
    return " ".join(" ".join(query).strip().split("\n"))
