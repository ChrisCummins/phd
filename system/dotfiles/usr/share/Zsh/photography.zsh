###############################################################################
# Overwrite the EXIF data of the given file(s) to match that of a
# Samyang lenses.
#
# If these files have already been imported to Lightroom, you should refresh
# the metadata cached in the Lightroom catalog by right clicking on the image
# and selecting "Metadata" > "Read Metadata from file".
#
# Usage:
#     $ samyang_{8,12}mm_set_exif <photo...>
#
samyang_12mm_set_exif() {
  exiftool -overwrite_original \
      -LensMake="Samyang" \
      -LensModel="12mm f/2.0 CS NCS" \
      -Lens="Samyang 12mm f/2.0" \
      -FocalLength="12" \
      -ApertureValue="2.0" \
      -FNumber="2.0" \
      $@
}

samyang_8mm_set_exif() {
  exiftool -overwrite_original \
      -LensMake="Samyang" \
      -LensModel="8mm f/2.8 UMC Fish-eye II" \
      -Lens="Samyang 8mm f/2.8" \
      -FocalLength="8" \
      -ApertureValue="2.8" \
      -FNumber="2.8" \
      $@
}

###############################################################################
