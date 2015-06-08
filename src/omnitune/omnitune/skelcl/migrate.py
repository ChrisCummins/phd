import labm8 as lab
from labm8 import db
from labm8 import fs
from labm8 import io

from omnitune.skelcl import db

def migrate_0_to_1(old):
    """
    SkelCL database migration script.

    Arguments:

        old (SkelCLDatabase): The database to migrate
    """
    def get_source(checksum):
        query = old.execute("SELECT source FROM kernels WHERE checksum = ?",
                            (checksum,))
        return query.fetchone()[0]

    def get_device_attr(device_id, name, count):
        query = old.execute("SELECT * FROM devices WHERE name = ?",
                            (name,))
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
        tmp.execute("INSERT OR IGNORE INTO kernels VALUES (?,?,?,?,?,?,?)",
                    (kernel_id,north,south,east,west,max_wg_size,user_source))

        placeholders = ",".join(["?"] * len(device_attr))
        tmp.execute("INSERT OR IGNORE INTO devices VALUES (" + placeholders + ")",
                    device_attr)

        tmp.execute("INSERT OR IGNORE INTO data VALUES (?,?,?,?,?)",
                    (data_id, data_width, data_height, type_in, type_out))

        tmp.execute("INSERT OR IGNORE INTO params VALUES (?,?,?)",
                    (params_id, wg_c, wg_r))

        tmp.execute("INSERT OR IGNORE INTO scenarios VALUES (?,?,?,?,?)",
                    (scenario_id, host, device_id, kernel_id, data_id))

        tmp.execute("INSERT INTO runtimes VALUES (?,?,?)",
                    (scenario_id, params_id, runtime))

    # Create temporary database
    tmp = db.Database("/tmp/omnitune.skelcl.migration.db")

    # Clear anything that's already in the database.
    for table in tmp.tables:
        tmp.drop_table(table)

    io.info("Migrating database to version 1.")

    backup_path = old.path + ".0"
    io.info("Creating backup of old database at '{0}'".format(backup_path))
    fs.cp(old.path, backup_path)

    io.debug("Migration: creating tables ...")

    # Create table: kernels
    tmp.create_table("version",
                     (("version",                         "integer"),))

    # Set database version
    tmp.execute("INSERT INTO version VALUES (1)")

    # Create table: kernels
    tmp.create_table("kernels",
                     (("id",                              "text primary key"),
                      ("north",                           "integer"),
                      ("south",                           "integer"),
                      ("east",                            "integer"),
                      ("west",                            "integer"),
                      ("max_wg_size",                     "integer"),
                      ("source",                          "text")))

    # Create table: devices
    tmp.create_table("devices",
                     (("id",                              "text primary key"),
                      ("name",                            "text"),
                      ("count",                           "integer"),
                      ("address_bits",                    "integer"),
                      ("double_fp_config",                "integer"),
                      ("endian_little",                   "integer"),
                      ("execution_capabilities",          "integer"),
                      ("extensions",                      "text"),
                      ("global_mem_cache_size",           "integer"),
                      ("global_mem_cache_type",           "integer"),
                      ("global_mem_cacheline_size",       "integer"),
                      ("global_mem_size",                 "integer"),
                      ("host_unified_memory",             "integer"),
                      ("image2d_max_height",              "integer"),
                      ("image2d_max_width",               "integer"),
                      ("image3d_max_depth",               "integer"),
                      ("image3d_max_height",              "integer"),
                      ("image3d_max_width",               "integer"),
                      ("image_support",                   "integer"),
                      ("local_mem_size",                  "integer"),
                      ("local_mem_type",                  "integer"),
                      ("max_clock_frequency",             "integer"),
                      ("max_compute_units",               "integer"),
                      ("max_constant_args",               "integer"),
                      ("max_constant_buffer_size",        "integer"),
                      ("max_mem_alloc_size",              "integer"),
                      ("max_parameter_size",              "integer"),
                      ("max_read_image_args",             "integer"),
                      ("max_samplers",                    "integer"),
                      ("max_work_group_size",             "integer"),
                      ("max_work_item_dimensions",        "integer"),
                      ("max_work_item_sizes_0",           "integer"),
                      ("max_work_item_sizes_1",           "integer"),
                      ("max_work_item_sizes_2",           "integer"),
                      ("max_write_image_args",            "integer"),
                      ("mem_base_addr_align",             "integer"),
                      ("min_data_type_align_size",        "integer"),
                      ("native_vector_width_char",        "integer"),
                      ("native_vector_width_double",      "integer"),
                      ("native_vector_width_float",       "integer"),
                      ("native_vector_width_half",        "integer"),
                      ("native_vector_width_int",         "integer"),
                      ("native_vector_width_long",        "integer"),
                      ("native_vector_width_short",       "integer"),
                      ("preferred_vector_width_char",     "integer"),
                      ("preferred_vector_width_double",   "integer"),
                      ("preferred_vector_width_float",    "integer"),
                      ("preferred_vector_width_half",     "integer"),
                      ("preferred_vector_width_int",      "integer"),
                      ("preferred_vector_width_long",     "integer"),
                      ("preferred_vector_width_short",    "integer"),
                      ("queue_properties",                "integer"),
                      ("single_fp_config",                "integer"),
                      ("type",                            "integer"),
                      ("vendor",                          "text"),
                      ("vendor_id",                       "text"),
                      ("version",                         "text")))

    # Create table: data
    tmp.create_table("data",
                     (("id",                              "text primary key"),
                      ("width",                           "integer"),
                      ("height",                          "integer"),
                      ("tin",                             "text"),
                      ("tout",                            "text")))

    # Create table: params
    tmp.create_table("params",
                     (("id",                              "text primary key"),
                      ("wg_c",                            "integer"),
                      ("wg_r",                            "integer")))

    # Create table: scenarios
    tmp.create_table("scenarios",
                     (("id",                              "text primary key"),
                      ("host",                            "text"),
                      ("device",                          "text"),
                      ("kernel",                          "text"),
                      ("data",                            "text")))

    # Create table: runtimes
    tmp.create_table("runtimes",
                     (("scenario",                        "text"),
                      ("params",                          "text"),
                      ("runtime",                         "real")))

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
    tmp = db.Database("/tmp/omnitune.skelcl.migration.db")

    io.info("Migrating database to version 2.")

    backup_path = old.path + ".1"
    io.info("Creating backup of old database at '{0}'".format(backup_path))
    fs.cp(old.path, backup_path)

    # Update database version
    tmp.drop_table("version")
    tmp.create_table("version",
                     (("version",                         "integer"),))
    tmp.execute("INSERT INTO version VALUES (2)")

    # Rename table "data" to "datasets"
    tmp.create_table("datasets",
                     (("id",                              "text primary key"),
                      ("width",                           "integer"),
                      ("height",                          "integer"),
                      ("tin",                             "text"),
                      ("tout",                            "text")))
    tmp.execute("INSERT INTO datasets SELECT * FROM data")
    tmp.drop_table("data")

    # Rename column "scenarios.data" to "scenarios.dataset"
    tmp.execute("ALTER TABLE scenarios RENAME TO old_scenarios")
    tmp.create_table("scenarios",
                     (("id",                              "text primary key"),
                      ("host",                            "text"),
                      ("device",                          "text"),
                      ("kernel",                          "text"),
                      ("dataset",                         "text")))
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
    if db.version == 0:
        migrate_0_to_1(db)
        db = db.Database()
    if db.version == 1:
        migrate_1_to_2(db)
        db = db.Database()

    return db
