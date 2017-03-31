#!/usr/bin/env python3
import progressbar

from argparse import ArgumentParser
from labm8 import fs
from tempfile import NamedTemporaryFile

import clinfo
import clsmith
import db
import re


def cl_launcher(src: str, platform_id: int, device_id: int, *args):
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.c') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return clsmith.cl_launcher(tmp.name, platform_id, device_id, *args)


def verify_platform(platform_name, stderr):
    for line in stderr.split('\n'):
        if line.startswith("Platform: "):
            actual_platform_name = re.sub(r"^Platform: ", "", line).rstrip()
            assert(actual_platform_name == platform_name)
            return


def verify_device(device_name, stderr):
    for line in stderr.split('\n'):
        if line.startswith("Device: "):
            actual_device_name = re.sub(r"^Device: ", "", line).rstrip()
            assert(actual_device_name == device_name)
            return


def get_params(stderr):
    global_size = None
    local_size = None
    for line in stderr.split('\n'):
        # global size
        match = re.match('^3-D global size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            global_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D global size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            global_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D global size \d+ = \[(\d+)\]', line)
        if match:
            global_size = (int(match.group(1)), 0, 0)

        # local size
        match = re.match('^3-D local size \d+ = \[(\d+), (\d+), (\d+)\]', line)
        if match:
            local_size = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        match = re.match('^2-D local size \d+ = \[(\d+), (\d+)\]', line)
        if match:
            local_size = (int(match.group(1)), int(match.group(2)), 0)
        match = re.match('^1-D local size \d+ = \[(\d+)\]', line)
        if match:
            local_size = (int(match.group(1)), 0, 0)

        if global_size and local_size:
            return global_size, local_size


def run_next_prog(platform, device, testbed_id):
    platform_id, platform_name = platform
    device_id, device_name = device

    with db.Session() as session:
        gsize = (128, 16, 1)
        lsize = (32, 1, 1)
        params = db.Params(
            gsize_x = gsize[0], gsize_y = gsize[1], gsize_z = gsize[2],
            lsize_x = lsize[0], lsize_y = lsize[1], lsize_z = lsize[2])
        nparam = session.query(db.Params).filter(
            db.Params.gsize_x == gsize[0],
            db.Params.gsize_y == gsize[1],
            db.Params.gsize_z == gsize[2],
            db.Params.lsize_x == lsize[0],
            db.Params.lsize_y == lsize[1],
            db.Params.lsize_z == lsize[2]
        ).count()
        if nparam == 0:
            session.add(params)
        session.flush()
        params = session.query(db.Params).filter(
            db.Params.gsize_x == gsize[0],
            db.Params.gsize_y == gsize[1],
            db.Params.gsize_z == gsize[2],
            db.Params.lsize_x == lsize[0],
            db.Params.lsize_y == lsize[1],
            db.Params.lsize_z == lsize[2]
        ).one()

        subquery = session.query(db.Result.program_id).filter(
            db.Result.testbed_id == testbed_id, db.Result.params_id == params.id)
        program = session.query(db.Program).filter(
            ~db.Program.id.in_(subquery)).order_by(db.Program.id).first()

        opts = params.to_args()
        runtime, status, stdout, stderr = cl_launcher(
            program.src, platform_id, device_id, *opts)

        # assert that run params match expected
        verify_platform(platform_name, stderr)
        verify_device(device_name, stderr)
        actual_gsize, actual_lsize = get_params(stderr)
        assert(gsize == actual_gsize)
        assert(lsize == actual_lsize)

        # add new result
        session.add(db.Result(
            program_id=program.id, testbed_id=testbed_id, params_id=params.id,
            cli=" ".join(opts), status=status, runtime=runtime,
            stdout=stdout, stderr=stderr))


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

    platform_name = clinfo.get_platform_name(platform_id)
    device_name = clinfo.get_device_name(args.platform_id, device_id)

    db.init(dbpath)  # initialize db engine

    testbed_id = db.register_testbed(platform_name, device_name)

    print('using testbed', testbed_id, 'using', device_name)

    ran, ntodo = db.get_num_progs_to_run(testbed_id)
    bar = progressbar.ProgressBar(initial_value=ran, max_value=ntodo)
    while True:
        run_next_prog((platform_id, platform_name), (device_id, device_name),
                      testbed_id)

        ran, ntodo = db.get_num_progs_to_run(testbed_id)
        bar.max_value = ntodo
        bar.update(ran)
        if not ntodo:
            break
