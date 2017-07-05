#!/usr/bin/env python

import datetime
import pyopencl as cl
import random
import os
import sys
from time import sleep

from argparse import ArgumentParser
from itertools import product
from subprocess import Popen

import util
import db
from db import *

from clsmith import *
from clgen_run_cl_launcher import *


def cl_launcher(src: str, platform_id: int, device_id: int,
                *args) -> Tuple[float, int, str, str]:
    """ Invoke cl launcher on source """
    with NamedTemporaryFile(prefix='cl_launcher-', suffix='.cl') as tmp:
        tmp.write(src.encode('utf-8'))
        tmp.flush()

        return clsmith.cl_launcher(tmp.name, platform_id, device_id, *args,
                                   timeout=os.environ.get("TIMEOUT", 60))


def reproduce(**args):
    TABLE = cl_launcherCLgenResult

    with Session(commit=False) as s:
        result = s.query(TABLE).filter(TABLE.id == args['result_id']).first()
        if not result:
            raise KeyError(f"no result with ID {args['result_id']}")

        flags = result.params.to_flags()
        program = result.program

        if args['report']:
            # generate bug report
            now = datetime.datetime.utcnow().isoformat()

            cli = ' '.join(cl_launcher_cli(
                "kernel.cl", '$PLATFORM_ID', '$DEVICE_ID', *flags,
                cl_launcher_path="./CLSmith/build/cl_launcher",
                include_path="./CLSmith/runtime/"))

            bug_type = {
                "w": "miscompilation",
                "c": "runtime crash"
            }[args['report']]

            print(f"""\
#!/usr/bin/env bash
set -eu

# {bug_type} bug report generated {now}
# execute this file to reproduce bug:
#    export PLATFORM_ID=<id>
#    export DEVICE_ID=<id>
#    $ bash ./bug-report.sh

# Metadata:
#   OpenCL Platform:        {result.testbed.platform}
#   OpenCL Device:          {result.testbed.device}
#   Driver version:         {result.testbed.driver}
#   OpenCL version:         {result.testbed.opencl}
#   Host Operating System:  {result.testbed.host}
""")
            if result.testbed.platform.lower().startswith("nvidia"):
                if not os.path.exists('nvidia-bug-report.log.gz'):
                    os.system('sudo nvidia-bug-report.sh > /dev/null')
                print("# see nvidia-bug-report.log.gz for more information")

            print(f"""
# Kernel:
cat << EOF > kernel.cl
{program.src}
EOF
echo "kernel written to 'kernel.cl'"

# Kernel parameters:
#   Global size: {result.params.gsize}
#   Workgroup size: {result.params.lsize}
#   Optimizations: {result.params.optimizations_on_off}
""")
            if args['report'] == "w":
                expected_output, majority_devs = util.get_majority_output(
                    s, result, cl_launcherCLgenResult)
                majority_dev_str = "\n".join([f"#   - {d.platform} {d.device}" for d in majority_devs])
                print(f"""
# Expected output:
cat << EOF > expected-output.txt
{expected_output}
EOF
echo "expected output written to 'expected-output.txt'"

# How was the expected output determined? Differential test
# Which devices computed a different result?
{majority_dev_str}

# Actual output:
cat << EOF > actual-output.txt
{result.stdout}
EOF
echo "actual output written to 'actual-output.txt'"
""")
            elif args['report'] == "c":
                print(f"""
# Program output:
cat << EOF > reported-stderr.txt
{result.stderr}
EOF
echo "reported output written to 'reported-stdout.txt' and 'reported-stderr.txt'"
echo "reported program returncode is {result.status}"
""")

            print(f"""# Build requirements (CLSmith):
if [ ! -d "./CLSmith" ]; then
    git clone https://github.com/ChrisCummins/CLSmith.git
    cd CLSmith
    git reset --hard b637b31c31e0f90ef199ca492af05172400df050
    cd ..
fi
if [ ! -d "./CLSmith/build" ]; then
    mkdir CLSmith/build
    cd CLSmith/build
    cmake ..
    make -j $(nproc)
    cp -v ../runtime/*.h .
    cd ../..
fi

# Run kernel using CLSmith's cl_launcher:
{cli} >stdout.txt 2>stderr.txt
echo "reproduced output written to 'stdout.txt' and 'stderr.txt'"
""")

            sys.exit(0)
        else:
            # lookup the device
            try:
                platform_id = result.testbed.platform_id()
                device_id = result.testbed.device_id()
            except KeyError as e:
                print(e, file=sys.stderr)
                sys.exit(1)

            # run the program
            runtime, status, stdout, stderr = cl_launcher(
                    program.src, platform_id, device_id, *flags)

            if verbose:
                print(stderr)
                print(stdout)

            reproduced = True
            if stderr != result.stderr:
                reproduced = False
                print("stderr differs")
            if stdout != result.stdout:
                reproduced = False
                print("stdout differs")

            sys.exit(0 if reproduced else 1)


def main():
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-r", "--result", dest="result_id", type=int, default=None,
                        help="results ID")
    parser.add_argument("--report",
                        help="generate bug report of type: {w,bc}")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # get testbed information
    db_hostname = args.hostname
    db_url = db.init(db_hostname)

    reproduce(**vars(args))

if __name__ == "__main__":
    main()
