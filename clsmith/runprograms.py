#!/usr/bin/env python3
import progressbar

from argparse import ArgumentParser
from labm8 import fs
from tempfile import NamedTemporaryFile

import clinfo
import clsmith
import db






def get_num_progs_to_run():
    pass


def get_next_program_id():
    with db.Session() as session:
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dbpath", metavar="<database>",
                        help="path to database")
    parser.add_argument("platform_id", metavar="<platform-id>", type=int,
                        help="OpenCL platform ID")
    parser.add_argument("device_id", metavar="<device-id>", type=int,
                        help="OpenCL device ID")
    args = parser.parse_args()

    platform_id = args.platform_id
    device_id = args.device_id
    dbpath = fs.path(args.dbpath)

    platform_name = clinfo.get_platform_name(args.platform_id)
    device_name = clinfo.get_device_name(args.platform_id, args.device_id)

    db.init(dbpath)  # initialize db engine

    testbed_id = db.register_testbed(platform_name, device_name)

    print('using testbed', testbed_id, 'using', device_name)
