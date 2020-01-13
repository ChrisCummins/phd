#!/usr/bin/env python3
#
# Copyright (C) 2015, 2016 Chris Cummins.
#
# This file is part of rt.
#
# rt is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# rt is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with rt.  If not, see <http://www.gnu.org/licenses/>.
from math import ceil
from os.path import abspath
from random import randint
from re import compile
from re import search
from re import sub
from sys import argv
from sys import exit
from sys import stdout


verbosity = {
  "debug": {"file_paths": True, "macro_defs": False},
  "warn": {"unused_def": True},
}


def debug(*args, **kwargs):
  print("[DEBUG ]", *args, **kwargs)


def warn(*args, **kwargs):
  print("[WARN  ]", *args, **kwargs)


def fatal(*args, **kwargs):
  print("[FATAL ]", *args, **kwargs)
  exit(1)


def tokenise(characters):
  # Parser state.
  in_quotes = False
  in_comment = False
  tokens = []
  buf = ""  # Token buffer.

  # Iterate over characters:
  for c in characters:
    # If it's a comment character, set "in_comment" flag.
    if c == "#":
      in_comment = True
    # If it's the end of a line, reset the "in_comment" flag, but
    # *don't* empty the buffer if we're in quotes, to support
    # multi-line strings.
    elif c == "\n" or c == "\r":
      in_comment = False
      if in_quotes:
        buf += c
      else:
        if buf:
          tokens.append(buf)
          buf = ""
      continue

    # If we're in a comment, do nothing.
    if in_comment:
      continue

    # If it's a quote character determine whether we're opening or
    # closing a quote. If we're closing, then dump the quoted
    # characters (excluding the surrounding "" quotation marks).
    if c == '"':
      if in_quotes:
        in_quotes = False
        if buf:
          tokens.append(buf)
          buf = ""
      else:
        in_quotes = True
        if buf:
          tokens.append(buf)
          buf = ""
    # If we're at a word break, dump the buffer, except if we're
    # in quotes.
    elif c == " " or c == "\t":
      if in_quotes:
        buf += c
      else:
        if buf:
          tokens.append(buf)
          buf = ""
    # Fall-through state: we're in a normal word.
    else:
      buf += c

  # Flush the last token (if any).
  if buf:
    tokens.append(buf)

  return tokens


# Return the absolute path to a file.
def get_path(path):
  return abspath(path)


# Accepts a line and returns the path to an @import statement, if
# found. Else, returns false.
def is_import_macro(line):
  match = search(import_re, line)

  if match:
    return get_path(match.group("path"))
  else:
    return False


def preprocess(tokens):
  return tokens


# Strip a line of comments and whitespace.
def strip_line(line):
  return sub(comment_re, "", line.strip())


def lookup_macro(word, macros):
  if word[0] != "@":
    return word

  key = word[1:]

  if key in macros:
    # Set "used" flag for word.
    macros[key][1] = True
    # Recurse based on key value.
    return lookup_macro(macros[key][0], macros)
  else:
    return word


def preprocess(tokens, macros={}):
  processed = []
  next_token_def = False
  next_token_def_val = False
  next_token_import = False

  for token in tokens:
    # First expand macros.
    token = lookup_macro(token, macros)
    lowertoken = token.lower()

    if next_token_import:
      # Expand @import statements.
      processed += get_tokens(token, macros)
      next_token_import = False
    elif next_token_def:
      # Set macro name.
      next_token_def_val = token
      next_token_def = False
    elif next_token_def_val:
      # Define new macro.
      name = next_token_def_val
      macros[name] = [token, False]
      if verbosity["debug"]["macro_defs"]:
        debug("Defined macro '{0}' = '{1}'".format(name, token))
      next_token_def_val = False
      next_token_def = False
    elif lowertoken == "@import":
      next_token_import = True
    elif lowertoken == "@def":
      next_token_def = True
    else:
      processed.append(token)

  return processed


def read_file(path):
  path = get_path(path)
  try:
    lines = open(path).readlines()
    if verbosity["debug"]["file_paths"]:
      debug("Read '{0}'".format(path))
    return lines
  except FileNotFoundError:
    fatal("No such file or directory: '{0}'.".format(path))


# Read an input file and return a list of source-tokens, stripped of
# comments, and recursively expanded @include statements and macros.
def get_tokens(path, macros={}):
  # Step 1. Read input file.
  lines = read_file(path)
  # Step 2. Tokenise.
  tokens = tokenise("".join(lines))
  # Step 3. Pre-process tokens.
  tokens = preprocess(tokens)

  return tokens


def get_sections(tokens):
  sections = []
  buf = []

  for token in tokens:
    if token[0] == "[" and token[-1] == "]":
      if buf:
        sections.append(buf)
      buf = [token[1:-1]]
    else:
      buf.append(token)

  if buf:
    sections.append(buf)

  return sections


colour_6_re = compile("^0x[0-9a-f]{6}$")


def get_colour(token):
  if search(colour_6_re, token):
    return "Colour({0})".format(token)
  else:
    fatal("Unrecognised colour: '{0}'".format(token))


def get_vector(tokens):
  return "Vector({0}, {1}, {2})".format(tokens[0], tokens[1], tokens[2])


def pairs_to_str(pairs):
  s = [key + ": " + " ".join(pairs[key]) for key in pairs]
  return '"' + ", ".join(s) + '"'


def percent_to_float(string):
  return float(string) / 100


def consume_colour(pairs, name, default="0x000"):
  if "colour" in pairs:
    val = get_colour("".join(pairs["colour"]))
    pairs.pop("colour", None)
    return val
  else:
    return get_colour(default)


def cast_scalar(val):
  return "static_cast<Scalar>({val})".format(val=val)


def consume_scalar(pairs, name, default=0):
  if name in pairs:
    val = float("".join(pairs[name]))
    pairs.pop(name, None)
  else:
    val = default
  return val


def consume_percent(pairs, name, default=0):
  if name in pairs:
    val = percent_to_float("".join(pairs[name]))
    pairs.pop(name, None)
    return val
  else:
    return default


def consume_int(pairs, name, default=0):
  if name in pairs:
    val = int("".join(pairs[name]))
    pairs.pop(name, None)
    return val
  else:
    return default


def consume_str(pairs, name, default=""):
  if name in pairs:
    val = "".join(pairs[name])
    pairs.pop(name, None)
    return val
  else:
    return default


def consume_vector(pairs, name, default=[0, 0, 0]):
  if name in pairs:
    val = get_vector(pairs[name])
    pairs.pop(name, None)
    return val
  else:
    return get_vector(default)


def consume_rgb(pairs, name, default=[100, 100, 100]):
  if name in pairs:
    val = pairs[name]
    rgb = [float(val[0]) / 100, float(val[1]) / 100, float(val[2]) / 100]
    pairs.pop(name, None)
    return rgb
  else:
    return get_vector(default)


material_name_re = "^\$[mM]aterial\.(?P<val>.+)"


def consume_material(pairs, name):
  if name in pairs:
    key = "".join(pairs[name])
    match = search(material_name_re, key)

    if not match:
      fatal("Invalid material name '{0}'".format(key))
    else:
      val = match.group("val")
      if val in materials:
        return val
      else:
        fatal("No material named '{0}'".format(val))

  else:
    fatal("Missing material name.")


renderer = {}


def set_renderer(pairs):
  renderer["depth"] = consume_int(pairs, "raydepth", default=100)
  renderer["scale"] = consume_int(pairs, "scale", default=1)
  renderer["dof"] = consume_int(pairs, "dofsamples", default=1)
  renderer["path"] = consume_str(pairs, "path", default="render.ppm")


def set_renderer_antialiasing(pairs):
  aa = {}
  renderer["aa"] = aa


def set_renderer_softlights(pairs):
  lights = {}
  renderer["lights"] = lights
  lights["base"] = consume_int(pairs, "base", default=3)
  lights["scalefactor"] = consume_scalar(pairs, "scalefactor", default=0.01)


materials = set()


def get_material_code(name, pairs):
  colour = consume_colour(pairs, "colour")
  ambient = consume_percent(pairs, "ambient")
  diffuse = consume_percent(pairs, "diffuse")
  specular = consume_percent(pairs, "specular")
  shininess = consume_int(pairs, "shininess")
  reflectivity = consume_percent(pairs, "reflectivity")

  if len(pairs):
    fatal("Unrecognised attributes:", pairs_to_str(pairs))

  if name in materials:
    fatal("Duplicate material name '{0}'".format(name))
  materials.add(name)

  return (
    "const Material *const restrict {name} = "
    "new Material({colour}, {ambient}, {diffuse}, "
    "{specular}, {shininess}, {reflectivity});".format(
      name=name,
      colour=colour,
      ambient=ambient,
      diffuse=diffuse,
      specular=specular,
      shininess=shininess,
      reflectivity=reflectivity,
    )
  )


objects = set()


def get_plane_code(name, pairs):
  position = consume_vector(pairs, "position")
  direction = consume_vector(pairs, "direction")
  material = consume_material(pairs, "material")

  if name in objects:
    fatal("Duplicate object name '{0}'".format(name))
  objects.add(name)

  return (
    "const Plane *const restrict {name} = "
    "new Plane({position}, {direction}, {material});".format(
      name=name, position=position, direction=direction, material=material
    )
  )


def get_checkerboard_code(name, pairs):
  position = consume_vector(pairs, "position")
  direction = consume_vector(pairs, "direction")
  size = consume_int(pairs, "size")
  material1 = consume_material(pairs, "material1")
  material2 = consume_material(pairs, "material2")

  if name in objects:
    fatal("Duplicate object name '{0}'".format(name))
  objects.add(name)

  return (
    "const CheckerBoard *const restrict {name} = "
    "new CheckerBoard({position}, {direction}, {size}, "
    "{material1}, {material2});".format(
      name=name,
      position=position,
      direction=direction,
      size=size,
      material1=material1,
      material2=material2,
    )
  )


def get_sphere_code(name, pairs):
  position = consume_vector(pairs, "position")
  size = consume_int(pairs, "size")
  material = consume_material(pairs, "material")

  if name in objects:
    fatal("Duplicate object name '{0}'".format(name))
  objects.add(name)

  return (
    "const Sphere *const restrict {name} = "
    "new Sphere({position}, {size}, {material});".format(
      name=name, position=position, size=size, material=material
    )
  )


lights = set()


def get_softlight_code(name, pairs):
  position = consume_vector(pairs, "position")
  colour = consume_colour(pairs, "colour")
  size = consume_int(pairs, "size")

  # Get renderer configuration. If the renderer configuration has
  # not been set, this should throw a fatal error.
  base = renderer["lights"]["base"]
  scale = renderer["lights"]["scalefactor"]

  # Calculate the number of samples:
  #
  #    N = Nb + (r*s)^3
  #
  # Where: Nb is the base number of samples.
  #        r  is the radius of the softlight.
  #        s  is the soft light scale factor.
  samples = ceil(base + (size * scale) ** 3)

  if name in lights:
    fatal("Duplicate light name '{0}'".format(name))
  lights.add(name)

  return (
    "const SoftLight *const restrict {name} = "
    "new SoftLight({position}, {colour}, {size}, {samples});".format(
      name=name, position=position, size=size, colour=colour, samples=samples
    )
  )


def get_pointlight_code(name, pairs):
  position = consume_vector(pairs, "position")
  colour = consume_colour(pairs, "colour")

  if name in lights:
    fatal("Duplicate light name '{0}'".format(name))
  lights.add(name)

  return (
    "const SoftLight *const restrict {name} = "
    "new SoftLight({position}, {colour});".format(
      name=name, position=position, colour=colour
    )
  )


def consume_val(pairs, name):
  val = pairs[name]
  pairs.pop(name, None)
  return "".join(val)


def consume_lens(pairs, name):
  val = sub("^\$[lL]ens\.", "", consume_val(pairs, name))
  return lenses[val]


def consume_film(pairs, name):
  val = sub("^\$[fF]ilm\.", "", consume_val(pairs, name))
  return films[val]


film = None
camera = None


def get_camera_perspective_code(name, pairs):
  global camera
  global film

  position = consume_vector(pairs, "position")
  lookat = consume_vector(pairs, "lookat")
  lens = consume_lens(pairs, "lens")
  film = consume_film(pairs, "film")

  if camera:
    fatal("Duplicate camera definitions: '{0}' and '{1}'".format(camera, name))
  camera = name

  return (
    "const Camera *const restrict {name} = "
    "new Camera({position}, {lookat}, "
    "{width}, {height}, "
    "Lens({focal}, {aperture}, {focus}));".format(
      name=name,
      position=position,
      lookat=lookat,
      width=film["width"],
      height=film["height"],
      focal=lens["focal"],
      aperture=lens["aperture"],
      focus=lens["focus"],
    )
  )


films = {}


def add_film(name, pairs):
  gamma = consume_rgb(pairs, "rgbgamma")
  saturation = consume_percent(pairs, "saturation", 100)
  width = consume_int(pairs, "width")
  height = consume_int(pairs, "height")

  if name in films:
    fatal("Duplicate film type '{0}'".format(name))
  films[name] = {
    "gamma": gamma,
    "saturation": saturation,
    "width": width,
    "height": height,
  }


lenses = {}


def add_lens(name, pairs):
  focal = consume_int(pairs, "focallength")
  aperture = consume_scalar(pairs, "aperture", 1)
  focus = consume_scalar(pairs, "focus", 1)

  if name in lenses:
    fatal("Duplicate lens type '{0}'".format(name))

  lenses[name] = {"focal": focal, "aperture": aperture, "focus": focus}


def get_keyval_pairs(tokens):
  pairs = {}
  key = ""

  for token in tokens:
    if token[-1] == ":":
      key = token[0:-1].lower()
      if key in pairs:
        fatal("Duplicate key '{0}'".format(key))
      pairs[key] = []
    else:
      if key:
        pairs[key].append(token)
      else:
        fatal("Value without key '{0}'".format(token))

  return pairs


def newid():
  return "__id{0:09d}__".format(randint(0, 999999999))


material_re = compile("^material\.")
film_re = compile("^film\.")
lens_re = compile("^lens\.")


def get_section_code(section):
  name = section[0].lower()
  pairs = get_keyval_pairs(section[1:])

  if name == "renderer":
    set_renderer(pairs)
  elif name == "renderer.antialiasing":
    set_renderer_antialiasing(pairs)
  elif name == "renderer.softlights":
    set_renderer_softlights(pairs)
  elif search(material_re, name):
    return get_material_code(sub(material_re, "", name), pairs)
  elif name == "object.plane":
    return get_plane_code(newid(), pairs)
  elif name == "object.checkerboard":
    return get_checkerboard_code(newid(), pairs)
  elif name == "object.sphere":
    return get_sphere_code(newid(), pairs)
  elif name == "light.soft":
    return get_softlight_code(newid(), pairs)
  elif name == "light.point":
    return get_pointlight_code(newid(), pairs)
  elif search(film_re, name):
    add_film(sub(film_re, "", name), pairs)
  elif search(lens_re, name):
    add_lens(sub(lens_re, "", name), pairs)
  elif name == "camera.perspective":
    return get_camera_perspective_code(newid(), pairs)
  else:
    return "// Not implemented: {0}".format(name)


def get_scene_code():
  c = "const Object *_objects[] = {\n"
  for object in objects:
    c += "  {0},\n".format(object)
  c += "};\n"
  c += "const Light *_lights[] = {\n"
  for light in lights:
    c += "  {0},\n".format(light)
  c += "};\n"
  c += "const Objects objects(_objects, _objects + (sizeof(_objects) / sizeof(_objects[0])));\n"
  c += "const Lights lights(_lights, _lights + (sizeof(_lights) / sizeof(_lights[0])));\n"
  c += "const Scene *const restrict scene = new Scene(objects, lights);\n"

  return c


def get_renderer_code():
  depth = renderer["depth"]
  dofsamples = renderer["dof"]

  c = (
    "Renderer *const renderer = new Renderer(*{scene}, {camera}, "
    "{dof}, {depth});".format(
      scene="scene", camera=camera, depth=depth, dof=dofsamples
    )
  )
  return c


def get_image():
  width = renderer["scale"] * film["width"]
  height = renderer["scale"] * film["height"]
  itype = "Image<{width}, {height}>".format(width=width, height=height)
  c = "{itype} *const image = new {itype}(" "{saturation}, {colour});".format(
    itype=itype,
    saturation=film["saturation"],
    colour=(
      "Colour({0}, {1}, {2})".format(
        film["gamma"][0], film["gamma"][1], film["gamma"][2]
      )
    ),
  )
  return {"code": c, "width": width, "height": height, "type": itype}


def get_code(sections):
  code = []
  render = []

  code.append('#include "rt/rt.h"')
  code.append("using namespace rt;")

  code.append("int main(int argc, char **argv) {")
  for section in sections:
    render.append(get_section_code(section))
  render.append(get_scene_code())
  render.append(get_renderer_code())
  [code.append(line) for line in render if line]

  image = get_image()
  code.append(image["code"])

  # Render code:
  code.append(
    'render<{itype}>(*renderer, "{path}", image);'.format(
      itype=image["type"], path=renderer["path"]
    )
  )
  code.append("return 0;")
  code.append("}")

  return "\n".join(code)


input = argv[1]
if len(argv) > 2:
  output = open(argv[2], "w")
else:
  output = stdout

out = output
tokens = get_tokens(input)
sections = get_sections(tokens)
code = get_code(sections)

print(code, file=out)
