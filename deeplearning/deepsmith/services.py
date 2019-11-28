# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
import pathlib
import socket

import grpc

from deeplearning.deepsmith.proto import service_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


class ServiceBase(object):
  def __init__(self, config: pbutil.ProtocolBuffer):
    self.config = config

  def __repr__(self):
    cls_name = type(self).__name__
    return (
      f"{cls_name}@{self.config.service.hostname}:"
      f"{self.config.service.port}"
    )


def AssertLocalServiceHostname(service_config: service_pb2.ServiceConfig):
  hostname = socket.gethostname()
  service_hostname = service_config.hostname
  if (
    service_hostname
    and service_hostname != "localhost"
    and service_hostname != hostname
  ):
    raise app.UsageError(
      f"System hostname {hostname} does not match service hostname "
      f"{service_hostname}"
    )


def AssertResponseStatus(status: service_pb2.ServiceStatus):
  if status.returncode != service_pb2.ServiceStatus.SUCCESS:
    app.Fatal(
      "Error! %s responded with status %s: %s",
      status.client,
      status.returncode,
      status.error_message,
    )


def BuildDefaultRequest(cls) -> pbutil.ProtocolBuffer:
  message = cls()
  message.status.client = socket.gethostname()
  return message


def BuildDefaultResponse(cls) -> pbutil.ProtocolBuffer:
  message = cls()
  message.status.client = socket.gethostname()
  message.status.returncode = service_pb2.ServiceStatus.SUCCESS
  return message


def ServiceConfigFromFlag(
  flag_name: str, service_config: pbutil.ProtocolBuffer
) -> pbutil.ProtocolBuffer:
  if not getattr(FLAGS, flag_name):
    raise app.UsageError(f"--{flag_name} not set.")
  config_path = pathlib.Path(getattr(FLAGS, flag_name))
  if not config_path.is_file():
    cls_name = type(service_config).__name__
    raise app.UsageError(f"{cls_name} file not found: '{config_path}'.")

  return pbutil.FromFile(config_path, service_config)


def GetServiceStub(service_config: service_pb2.ServiceConfig, service_stub_cls):
  address = (
    f"{service_config.service.hostname}:" f"{service_config.service.port}"
  )
  channel = grpc.insecure_channel(address)
  stub = service_stub_cls(channel)
  app.Log(1, f"Connected to {service_stub_cls.__name__} at {address}")
  return stub
