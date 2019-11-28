from omnitune.skelcl import db as _db

from . import hash_dataset
from . import hash_device
from . import hash_scenario
from labm8.db import placeholders
from labm8.py import fs
from labm8.py import io


def migrate_0_to_1(old):
  """
  SkelCL database migration script.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """

  def get_source(checksum):
    query = old.execute(
      "SELECT source FROM kernels WHERE checksum = ?", (checksum,)
    )
    return query.fetchone()[0]

  def get_device_attr(device_id, name, count):
    query = old.execute("SELECT * FROM devices WHERE name = ?", (name,))
    attr = query.fetchone()

    # Splice into the new
    newattr = (device_id, attr[0], count) + attr[2:]
    return newattr

  def process_row(tmp, row):
    # Get column values from row.
    host = row[0]
    dev_name = row[1]
    dev_count = row[2]
    kern_checksum = row[3]
    north = row[4]
    south = row[5]
    east = row[6]
    west = row[7]
    data_width = row[8]
    data_height = row[9]
    max_wg_size = row[10]
    wg_c = row[11]
    wg_r = row[12]
    runtime = row[13]
    type_in = "float"
    type_out = "float"

    # Lookup source code.
    source = get_source(kern_checksum)
    user_source = get_user_source(source)

    kernel_id = hash_kernel(north, south, east, west, max_wg_size, source)
    device_id = hash_device(dev_name, dev_count)
    data_id = hash_data(data_width, data_height, type_in, type_out)
    scenario_id = hash_scenario(host, device_id, kernel_id, data_id)
    params_id = hash_workgroup_size(wg_c, wg_r)

    device_attr = get_device_attr(device_id, dev_name, dev_count)

    # Add database entries.
    tmp.execute(
      "INSERT OR IGNORE INTO kernels VALUES (?,?,?,?,?,?,?)",
      (kernel_id, north, south, east, west, max_wg_size, user_source),
    )

    placeholders = ",".join(["?"] * len(device_attr))
    tmp.execute(
      "INSERT OR IGNORE INTO devices VALUES (" + placeholders + ")", device_attr
    )

    tmp.execute(
      "INSERT OR IGNORE INTO data VALUES (?,?,?,?,?)",
      (data_id, data_width, data_height, type_in, type_out),
    )

    tmp.execute(
      "INSERT OR IGNORE INTO params VALUES (?,?,?)", (params_id, wg_c, wg_r)
    )

    tmp.execute(
      "INSERT OR IGNORE INTO scenarios VALUES (?,?,?,?,?)",
      (scenario_id, host, device_id, kernel_id, data_id),
    )

    tmp.execute(
      "INSERT INTO runtimes VALUES (?,?,?)", (scenario_id, params_id, runtime)
    )

  # Create temporary database
  tmp = _db.Database("/tmp/omnitune.skelcl.migration.db")

  # Clear anything that's already in the database.
  for table in tmp.tables:
    tmp.drop_table(table)

  io.info("Migrating database to version 1.")

  backup_path = old.path + ".0"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(old.path, backup_path)

  io.debug("Migration: creating tables ...")

  # Create table: kernels
  tmp.create_table("version", (("version", "integer"),))

  # Set database version
  tmp.execute("INSERT INTO version VALUES (1)")

  # Create table: kernels
  tmp.create_table(
    "kernels",
    (
      ("id", "text primary key"),
      ("north", "integer"),
      ("south", "integer"),
      ("east", "integer"),
      ("west", "integer"),
      ("max_wg_size", "integer"),
      ("source", "text"),
    ),
  )

  # Create table: devices
  tmp.create_table(
    "devices",
    (
      ("id", "text primary key"),
      ("name", "text"),
      ("count", "integer"),
      ("address_bits", "integer"),
      ("double_fp_config", "integer"),
      ("endian_little", "integer"),
      ("execution_capabilities", "integer"),
      ("extensions", "text"),
      ("global_mem_cache_size", "integer"),
      ("global_mem_cache_type", "integer"),
      ("global_mem_cacheline_size", "integer"),
      ("global_mem_size", "integer"),
      ("host_unified_memory", "integer"),
      ("image2d_max_height", "integer"),
      ("image2d_max_width", "integer"),
      ("image3d_max_depth", "integer"),
      ("image3d_max_height", "integer"),
      ("image3d_max_width", "integer"),
      ("image_support", "integer"),
      ("local_mem_size", "integer"),
      ("local_mem_type", "integer"),
      ("max_clock_frequency", "integer"),
      ("max_compute_units", "integer"),
      ("max_constant_args", "integer"),
      ("max_constant_buffer_size", "integer"),
      ("max_mem_alloc_size", "integer"),
      ("max_parameter_size", "integer"),
      ("max_read_image_args", "integer"),
      ("max_samplers", "integer"),
      ("max_work_group_size", "integer"),
      ("max_work_item_dimensions", "integer"),
      ("max_work_item_sizes_0", "integer"),
      ("max_work_item_sizes_1", "integer"),
      ("max_work_item_sizes_2", "integer"),
      ("max_write_image_args", "integer"),
      ("mem_base_addr_align", "integer"),
      ("min_data_type_align_size", "integer"),
      ("native_vector_width_char", "integer"),
      ("native_vector_width_double", "integer"),
      ("native_vector_width_float", "integer"),
      ("native_vector_width_half", "integer"),
      ("native_vector_width_int", "integer"),
      ("native_vector_width_long", "integer"),
      ("native_vector_width_short", "integer"),
      ("preferred_vector_width_char", "integer"),
      ("preferred_vector_width_double", "integer"),
      ("preferred_vector_width_float", "integer"),
      ("preferred_vector_width_half", "integer"),
      ("preferred_vector_width_int", "integer"),
      ("preferred_vector_width_long", "integer"),
      ("preferred_vector_width_short", "integer"),
      ("queue_properties", "integer"),
      ("single_fp_config", "integer"),
      ("type", "integer"),
      ("vendor", "text"),
      ("vendor_id", "text"),
      ("version", "text"),
    ),
  )

  # Create table: data
  tmp.create_table(
    "data",
    (
      ("id", "text primary key"),
      ("width", "integer"),
      ("height", "integer"),
      ("tin", "text"),
      ("tout", "text"),
    ),
  )

  # Create table: params
  tmp.create_table(
    "params",
    (("id", "text primary key"), ("wg_c", "integer"), ("wg_r", "integer")),
  )

  # Create table: scenarios
  tmp.create_table(
    "scenarios",
    (
      ("id", "text primary key"),
      ("host", "text"),
      ("device", "text"),
      ("kernel", "text"),
      ("data", "text"),
    ),
  )

  # Create table: runtimes
  tmp.create_table(
    "runtimes", (("scenario", "text"), ("params", "text"), ("runtime", "real"))
  )

  i = 0
  for row in old.execute("SELECT * from runtimes"):
    process_row(tmp, row)
    i += 1
    if not i % 2500:
      io.debug("Processed", i, "rows ...")
      if not i % 5000:
        tmp.commit()

  tmp.commit()

  old_path = old.path
  tmp_path = tmp.path

  # Copy migrated database over the original one.
  fs.cp(tmp_path, old_path)
  fs.rm(tmp_path)

  old.close()
  tmp.close()
  io.info("Migration completed.")


def migrate_1_to_2(old):
  """
  SkelCL database migration script.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """
  # Create temporary database
  fs.cp(old.path, "/tmp/omnitune.skelcl.migration.db")
  tmp = _db.Database("/tmp/omnitune.skelcl.migration.db")

  io.info("Migrating database to version 2.")

  backup_path = old.path + ".1"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(old.path, backup_path)

  # Update database version
  tmp.drop_table("version")
  tmp.create_table("version", (("version", "integer"),))
  tmp.execute("INSERT INTO version VALUES (2)")

  # Rename table "data" to "datasets"
  tmp.create_table(
    "datasets",
    (
      ("id", "text primary key"),
      ("width", "integer"),
      ("height", "integer"),
      ("tin", "text"),
      ("tout", "text"),
    ),
  )
  tmp.execute("INSERT INTO datasets SELECT * FROM data")
  tmp.drop_table("data")

  # Rename column "scenarios.data" to "scenarios.dataset"
  tmp.execute("ALTER TABLE scenarios RENAME TO old_scenarios")
  tmp.create_table(
    "scenarios",
    (
      ("id", "text primary key"),
      ("host", "text"),
      ("device", "text"),
      ("kernel", "text"),
      ("dataset", "text"),
    ),
  )
  tmp.execute("INSERT INTO scenarios SELECT * FROM old_scenarios")
  tmp.drop_table("old_scenarios")

  tmp.commit()

  old_path = old.path
  tmp_path = tmp.path

  # Copy migrated database over the original one.
  fs.cp(tmp_path, old_path)
  fs.rm(tmp_path)

  old.close()
  tmp.close()
  io.info("Migration completed.")


def migrate_2_to_3(old):
  """
  SkelCL database migration script.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """

  def _old_kernel2new(old_id):
    kernel = old.execute(
      "SELECT north,south,east,west,max_wg_size,source "
      "FROM kernels WHERE id=?",
      (old_id,),
    ).fetchone()
    if kernel:
      return tmp.kernel_id(*kernel)

  def _old_scenario2new(old_id):
    device, old_kernel, dataset = old.execute(
      "SELECT device,kernel,dataset " "FROM scenarios WHERE id=?", (old_id,)
    ).fetchone()
    kernel = _old_kernel2new(old_kernel)
    return tmp.scenario_id(device, kernel, dataset)

  # TODO: Un-comment out code!

  # Create temporary database
  fs.rm("/tmp/omnitune.skelcl.migration.db")
  tmp = _db.Database("/tmp/omnitune.skelcl.migration.db")
  tmp.attach(old.path, "rhs")

  io.info("Migrating database to version 3.")

  backup_path = old.path + ".2"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(old.path, backup_path)

  tmp_path = tmp.path
  old_path = old.path

  tmp.run("create_tables")

  # Populate feature and lookup tables.
  for row in old.execute("SELECT * FROM devices"):
    features = row[1:]
    id = hash_device(*features)
    io.debug("Features extracted for device", id)
    row = (id,) + features
    tmp.execute("INSERT INTO devices VALUES " + placeholders(*row), row)

    row = (features[0], features[1], id)
    tmp.execute("INSERT INTO device_lookup VALUES " + placeholders(*row), row)
    tmp.commit()

  for row in old.execute("SELECT * FROM kernels"):
    args = row[1:]
    tmp.kernel_id(*args)

  for row in old.execute("SELECT * FROM datasets"):
    features = row[1:]
    id = hash_dataset(*features)
    io.debug("Features extracted for dataset", id)
    row = (id,) + features
    tmp.execute("INSERT INTO datasets VALUES " + placeholders(*row), row)

    row = features + (id,)
    tmp.execute("INSERT INTO dataset_lookup VALUES " + placeholders(*row), row)
    tmp.commit()

  # Populate kernel_names table.
  for row in old.execute("SELECT * FROM kernel_names"):
    old_id = row[0]
    synthetic, name = row[1:]

    kernel = _old_kernel2new(old_id)
    if kernel:
      row = (kernel, synthetic, name)
      tmp.execute(
        "INSERT OR IGNORE INTO kernel_names VALUES " + placeholders(*row), row
      )
  tmp.commit()

  # Populate scenarios table.
  for row in old.execute("SELECT * FROM scenarios"):
    old_id, _, device, old_kernel, dataset = row
    kernel = _old_kernel2new(old_kernel)
    new_id = hash_scenario(device, kernel, dataset)

    row = (new_id, device, kernel, dataset)
    tmp.execute(
      "INSERT OR IGNORE INTO scenarios VALUES " + placeholders(*row), row
    )
  tmp.commit()

  # Populate params table.
  tmp.execute("INSERT INTO params SELECT * from rhs.params")
  tmp.commit()

  scenario_replacements = {
    row[0]: _old_scenario2new(row[0])
    for row in old.execute("SELECT * FROM scenarios")
  }

  tmp.execute("INSERT INTO runtimes SELECT * from rhs.runtimes")
  for old_id, new_id in scenario_replacements.iteritems():
    io.info("Runtimes", old_id, "->", new_id)
    tmp.execute(
      "UPDATE runtimes SET scenario=? WHERE scenario=?", (new_id, old_id)
    )
  tmp.commit()

  # Sanity checks
  bad = False
  for row in tmp.execute("SELECT DISTINCT scenario FROM runtimes"):
    count = tmp.execute(
      "SELECT Count(*) FROM scenarios WHERE id=?", (row[0],)
    ).fetchone()[0]
    if count != 1:
      io.error("Bad scenario count:", row[0], count)
      bad = True

  if bad:
    io.fatal("Failed sanity check, aborting.")
  else:
    io.info("Passed sanity check.")

  # Copy migrated database over the original one.
  fs.cp(tmp_path, old_path)
  fs.rm(tmp_path)

  old.close()
  tmp.close()
  io.info("Migration completed.")


def migrate_3_to_4(old):
  """
  SkelCL database migration script.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """
  # Create temporary database
  fs.rm("/tmp/omnitune.skelcl.migration.db")
  tmp = _db.Database("/tmp/omnitune.skelcl.migration.db")
  tmp.attach(old.path, "rhs")

  io.info("Migrating database to version 4.")

  backup_path = old.path + ".3"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(old.path, backup_path)

  tables = [
    "kernels",
    "kernel_lookup",
    "kernel_names",
    "devices",
    "device_lookup",
    "datasets",
    "dataset_lookup",
    "scenarios",
    "params",
    "runtimes",
    "runtime_stats",
    "oracle_params",
  ]

  for table in tables:
    io.info("Copying data from '{}' ...".format(table))
    tmp.execute("INSERT INTO {} SELECT * FROM rhs.{}".format(table, table))

  tmp_path = tmp.path
  old_path = old.path

  tmp.execute("VACUUM")

  # Sanity checks
  bad = False
  for table in tables:
    old_count = tmp.num_rows("rhs." + table)
    tmp_count = tmp.num_rows(table)

    if old_count != tmp_count:
      io.error("Bad rows count:", old_count, tmp_count)
      bad = True

  if bad:
    io.fatal("Failed sanity check, aborting.")
  else:
    io.info("Passed sanity check.")

  # Copy migrated database over the original one.
  fs.cp(tmp_path, old_path)
  fs.rm(tmp_path)

  old.close()
  tmp.close()
  io.info("Migration completed.")


def migrate_4_to_5(db):
  """
  SkelCL database migration script.

  Database version 5 adds an additional "param_stats" table.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """
  io.info("Migrating database to version 5.")

  backup_path = db.path + ".4"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(db.path, backup_path)

  db.execute("DELETE FROM version")
  db.execute("INSERT INTO version VALUES (5)")

  db.execute(
    """
-- Parameter stats table
CREATE TABLE IF NOT EXISTS param_stats (
    params                          VARCHAR(255), -- Key for params
    num_scenarios                   INTEGER,      -- Number of scenarios for which param is legal, 0 < num_scenarios
    coverage                        REAL,         -- num_scenarios / total number of scenarios, 0 < coverage <= 1
    performance                     REAL,         -- Geometric mean of performance relative to the oracle for all scenarios for which param was legal, 0 < performance <= 1
    PRIMARY KEY (params)
)
"""
  )

  db.populate_param_stats_table()

  # Sanity checks
  bad = False
  if db.num_rows("param_stats") != len(db.params):
    io.error(
      "Bad row count in params table! Expected",
      len(db.params),
      "Observed:",
      db.num_rows("param_stats"),
    )
    bad = True

  if bad:
    io.fatal("Failed sanity check, aborting.")
  else:
    io.info("Passed sanity check.")

  # Copy migrated database over the original one.
  db.close()
  io.info("Migration completed.")


def migrate_5_to_6(db):
  """
  SkelCL database migration script.

  Database version 5 adds an additional "param_stats" table.

  Arguments:

      old (SkelCLDatabase): The database to migrate
  """
  io.info("Migrating database to version 6.")

  backup_path = db.path + ".5"
  io.info("Creating backup of old database at '{0}'".format(backup_path))
  fs.cp(db.path, backup_path)

  db.execute("DELETE FROM version")
  db.execute("INSERT INTO version VALUES (6)")

  db.execute(
    """
CREATE TABLE IF NOT EXISTS scenario_stats (
    scenario                        CHAR(40),     -- Key for scenarios
    num_params                      INTEGER,      -- The number of parameters in W_legal for scenario
    oracle_param                    VARCHAR(255), -- The best parameter
    oracle_runtime                  REAL,         -- The runtime of the best parameter
    worst_param                     VARCHAR(255), -- The worst parameter
    worst_runtime                   REAL,         -- The runtime of the worst parameter
    mean_runtime                    REAL,         -- The mean runtime of all parameters
    PRIMARY KEY (scenario)
)
"""
  )

  db.populate_scenario_stats_table()

  # Sanity checks
  bad = False
  if db.num_rows("scenario_stats") != len(db.scenarios):
    io.error(
      "Bad row count in scenario_stats table! Expected",
      len(db.scenarios),
      "Observed:",
      db.num_rows("scenario_stats"),
    )
    bad = True

  if bad:
    io.fatal("Failed sanity check, aborting.")
  else:
    io.info("Passed sanity check.")

  # Copy migrated database over the original one.
  db.close()
  io.info("Migration completed.")


def migrate(db):
  """
  Perform database migration.

  Migrates databases to the current version. If no migration is
  required, nothing happens.

  Arguments:

      db (skelcl.Database): Database to migrate

  Returns:

      skelcl.Database: Migrated database
  """
  path = db.path
  if db.version == 0:
    migrate_0_to_1(db)
    db = migrate(_db.Database(path=path))
  if db.version == 1:
    migrate_1_to_2(db)
    db = migrate(_db.Database(path=path))
  if db.version == 2:
    migrate_2_to_3(db)
    db = migrate(_db.Database(path=path))
  if db.version == 3:
    migrate_3_to_4(db)
    db = migrate(_db.Database(path=path))
  if db.version == 4:
    migrate_4_to_5(db)
    db = migrate(_db.Database(path=path))
  if db.version == 5:
    migrate_5_to_6(db)
    db = migrate(_db.Database(path=path))

  return db
