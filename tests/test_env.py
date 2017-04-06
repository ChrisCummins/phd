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
from unittest import TestCase, skip, main

import pyopencl as cl

import cldrive


class TestEnv(TestCase):
    def test_make_env_not_found(self):
        with self.assertRaises(LookupError):
            cldrive.make_env(platform_id=9999999, device_id=9999999)

    @skip("segfault when creating multiple envs")
    def test_make_env_cpu(self):
        env = cldrive.make_env(devtype="cpu")
        device = env.queue.get_info(cl.command_queue_info.DEVICE)
        device_type = device.get_info(cl.device_info.TYPE)
        self.assertEqual(device_type, cl.device_type.CPU)

    @skip("segfault when creating multiple envs")
    def test_make_env_gpu(self):
        env = cldrive.make_env(devtype="gpu")
        device = env.queue.get_info(cl.command_queue_info.DEVICE)
        device_type = device.get_info(cl.device_info.TYPE)
        self.assertEqual(device_type, cl.device_type.GPU)


if __name__ == "__main__":
    main()
