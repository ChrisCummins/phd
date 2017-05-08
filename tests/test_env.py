# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
import pytest

import cldrive

from lib import *


def test_make_env_not_found():
    with pytest.raises(LookupError):
        cldrive.make_env(platform="not a real plat",
                         device="not a real dev")


def test_all_envs():
    min_num_devices = 0

    if cldrive.has_cpu(): min_num_devices += 1
    if cldrive.has_gpu(): min_num_devices += 1

    envs = list(cldrive.all_envs())
    assert len(envs) >= min_num_devices


@needs_cpu
def test_make_env_cpu():
    env = cldrive.make_env(devtype="cpu")
    assert env.device_type == "CPU"


@needs_gpu
def test_make_env_gpu():
    env = cldrive.make_env(devtype="gpu")
    assert env.device_type == "GPU"
