"""Mark all-except-one duplicate bytecodes as "failed"."""
import sqlalchemy as sql

from deeplearning.ml4pl.bytecode import bytecode_database
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()

  with bytecode_db.Session() as session:
    sq = session.query(
        bytecode_database.LlvmBytecode.bytecode_sha1,
        sql.func.count(bytecode_database.LlvmBytecode.bytecode_sha1).label('count'))\
      .filter(bytecode_database.LlvmBytecode.clang_returncode == 0)\
      .group_by(bytecode_database.LlvmBytecode.bytecode_sha1)\
      .subquery()

    bytecodes_to_process = session.query(sq.c.bytecode_sha1, sq.c.count)\
      .filter(sq.c.count > 1)\
      .order_by(sq.c.count.desc())

    to_process_count = bytecodes_to_process.count()

    for i, (bytecode_sha1, duplicate_count) in enumerate(bytecodes_to_process):
      app.Log(1, 'Processing %s of %s, %s duplicates', i + 1, to_process_count,
              duplicate_count)

      update = sql.update(bytecode_database.LlvmBytecode) \
        .where(bytecode_database.LlvmBytecode.bytecode_sha1 == bytecode_sha1) \
        .values(clang_returncode=1,
                error_message=f'Duplicate LLVM module ({duplicate_count} duplicates)')
      bytecode_db.engine.execute(update)

      to_update = session.query(bytecode_database.LlvmBytecode.id) \
        .filter(bytecode_database.LlvmBytecode.bytecode_sha1 == bytecode_sha1) \
        .limit(1).one()
      update = sql.update(bytecode_database.LlvmBytecode) \
        .values(clang_returncode=0) \
        .where(bytecode_database.LlvmBytecode.id == to_update)
      bytecode_db.engine.execute(update)

      if not i % 512:
        app.Log(1, 'Committing')
        session.commit()

  app.Log(1, 'Done')


if __name__ == '__main__':
  app.Run(main)
