"""
SQL schema for FreeFocus.
"""
import sqlalchemy as sqlalchemy

from util.freefocus import sql

make_session = None


class System:

  @staticmethod
  def helo():
    global make_session

    engine = sqlalchemy.create_engine(args.uri, echo=args.verbose)

    sql.Base.metadata.create_all(engine)
    sql.Base.metadata.bind = engine
    make_session = sqlalchemy.orm.sessionmaker(bind=engine)

    return {"spec": freefocus.__freefocus_spec__}

  @staticmethod
  def new_person(**fields):
    q = session.query(Person) \
      .filter(Person.name == github_user.login,
              Person.created == github_user.created_at)

    if github_user.email:
      q = q.join(Email) \
        .filter(Email.address == github_user.email)

    if q.count():
      return q.one()
    else:
      p = Person(name=github_user.login, created=github_user.created_at)

      if github_user.email:
        p.emails.append(Email(address=github_user.email))

      session.add(p)
      get_or_create_usergroup(p)  # create group
      session.commit()

      return p
