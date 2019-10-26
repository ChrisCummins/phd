"""Module for splitting bytecode dataset into groups."""
import numpy as np
import random
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.bytecode import bytecode_database
from labm8 import app
from labm8 import humanize
from labm8 import prof

FLAGS = app.FLAGS

app.DEFINE_list('train_val_test_ratio', [3, 1, 1],
                'The ratio of training to validation to test dataset sizes.')
app.DEFINE_string('bytecode_split_type', 'all',
                  'The name of the bytecode dataset split to use.')
app.DEFINE_integer(
    'max_bytecode_split_size', 0,
    'The maximum number of bytecodes to include in a splt. If '
    '0, there is no limit.')


def GetTrainValTestGroups(
    db: bytecode_database.Database,
    train_val_test_ratio: typing.Iterable[float] = (3, 1, 1)
) -> typing.Dict[str, typing.List[int]]:
  """Get the bytecode IDs split into train, val, and test groups.

  This concatenates the POJ-104 sources with the other sources split into
  train/val/test segments.

  Args:
    db: The database to read IDs from.
    train_val_test_ratio: A triplet of ratios for the training, validation, and
      test sets. E.g. with the triplet (3, 1, 1), the training set will be 3/5
      of the dataset, and the validation and test sets will each by 1/5 of the
      dataset.

  Returns:
    A dictionary of bytecode IDs with "train", "val", and "test" keys.
  """
  # Normalize the ratios to sum to 1.
  ratios = np.array(list(train_val_test_ratio), dtype=np.float32)
  ratios /= sum(ratios)

  with db.Session() as s:
    poj104 = GetPoj104BytecodeGroups(db)

    # Get the bytecode IDs of non-POJ-104
    num_bytecodes = s.query(sql.func.count(bytecode_database.LlvmBytecode.id))\
      .filter(~bytecode_database.LlvmBytecode.source_name.like('poj-104:%'))\
      .one()[0]
    train_val_test_counts = np.floor(ratios * num_bytecodes).astype(np.int32)
    total_count = train_val_test_counts.sum()
    app.Log(1, 'Splitting %s bytecodes into groups: %s train, %s val, %s test',
            humanize.Commas(total_count + sum(len(s) for s in poj104.values())),
            humanize.Commas(train_val_test_counts[0] + len(poj104['train'])),
            humanize.Commas(train_val_test_counts[1] + len(poj104['val'])),
            humanize.Commas(train_val_test_counts[2] + len(poj104['test'])))

    q = s.query(bytecode_database.LlvmBytecode.id) \
      .filter(~bytecode_database.LlvmBytecode.source_name.like('poj-104:%'))\
      .order_by(db.Random())
    ids = [r[0] for r in q]

  return ApplySplitSizeLimit({
      'train':
      ids[:train_val_test_counts[0]] + poj104['train'],
      'val': (ids[train_val_test_counts[0]:sum(train_val_test_counts[:2])] +
              poj104['val']),
      'test':
      ids[sum(train_val_test_counts[:2]):] + poj104['test'],
  })


def GetPoj104BytecodeGroups(db: bytecode_database.Database,
                           ) -> typing.Dict[str, typing.List[int]]:
  """Get the bytecode IDs for the POJ-104 app classification experiment."""

  def GetBytecodeIds(filter_cb) -> typing.List[int]:
    """Return the bytecode IDs from the given filtered query."""
    with db.Session() as session:
      q = session.query(bytecode_database.LlvmBytecode.id).filter(filter_cb())
      return [r[0] for r in q]

  train = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:train'
  test = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:test'
  val = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:val'
  return ApplySplitSizeLimit({
      "train": GetBytecodeIds(train),
      "val": GetBytecodeIds(val),
      "test": GetBytecodeIds(test),
  })


def ApplySplitSizeLimit(groups: typing.Dict[str, typing.List[int]]):
  """Limit the size of ID lists per group if --max_bytecode_split_size > 0.

  Ags:
    groups: A mapping of group name to IDs.

  Returns:
    The groups dictionary, where each ID list has been limited to
    --max_bytecode_split_size elements, if --max_bytecode_split_size > 0. Else,
    the groups are returned unmodified.
  """
  if FLAGS.max_bytecode_split_size:
    for group in groups:
      random.shuffle(groups[group])
      groups[group] = groups[group][:FLAGS.max_bytecode_split_size]

  return groups


def GetGroupsFromFlags(
    db: bytecode_database) -> typing.Dict[str, typing.List[int]]:
  """Get bytecode ID groups using the flags.

  Args:
    db: The database to get the groups from.

  Returns:
     A dictionary of group names to list of bytecode IDs.
  """
  with prof.Profile(f'Read {FLAGS.bytecode_split_type} groups from database'):
    train_val_test_ratio = [float(x) for x in FLAGS.train_val_test_ratio]
    if FLAGS.bytecode_split_type == 'all':
      return GetTrainValTestGroups(db, train_val_test_ratio)
    elif FLAGS.bytecode_split_type == 'poj104':
      return GetPoj104BytecodeGroups(db)
