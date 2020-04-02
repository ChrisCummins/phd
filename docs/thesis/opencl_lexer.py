# -*- coding: utf-8 -*-
"""Pygments lexer for OpenCL."""
from pygments.lexers.c_cpp import CLexer
from pygments.token import Keyword
from pygments.token import Name

__all__ = ["OpenCLLexer"]


class OpenCLLexer(CLexer):
  """Pygments Lexer for OpenCL."""

  name = "OpenCL"
  aliases = ["opencl"]
  filenames = ["*.cl"]
  mimetypes = ["text/x-opencl-src"]

  keywords = set(
    (
      "kernel",
      "global",
      "local",
      "constant",
      "private",
      "read_only",
      "write_only",
    )
  )
  types = set(
    (
      "char2",
      "char3",
      "char4",
      "char8",
      "char16",
      "uchar2",
      "uchar3",
      "uchar4",
      "uchar8",
      "uchar16",
      "short2",
      "short3",
      "short4",
      "short8",
      "short16",
      "ushort2",
      "ushort3",
      "ushort4",
      "ushort8",
      "ushort16",
      "int2",
      "int3",
      "int4",
      "int8",
      "int16",
      "uint2",
      "uint3",
      "uint4",
      "uint8",
      "uint16",
      "long2",
      "long3",
      "long4",
      "long8",
      "long16",
      "ulong2",
      "ulong3",
      "ulong4",
      "ulong8",
      "ulong16",
      "float2",
      "float3",
      "float4",
      "float8",
      "float16",
      "double2",
      "double3",
      "double4",
      "double8",
      "double16",
      "image2d_t",
      "image3d_t",
      "sampler_t",
      "event_t",
      "size_t",
      "bool2",
      "bool3",
      "bool4",
      "bool8",
      "bool16",
      "half2",
      "half3",
      "half4",
      "half8",
      "half16",
      "quad",
      "quad2",
      "quad3",
      "quad4",
      "quad8",
      "quad16",
      "complex",
      "imaginary",
      "barrier",
    )
  )

  def get_tokens_unprocessed(self, text):
    for index, token, value in CLexer.get_tokens_unprocessed(self, text):
      if token is Name:
        if value in self.keywords:
          token = Keyword.Keyword
        elif value in self.types:
          token = Keyword.Type
      yield index, token, value
