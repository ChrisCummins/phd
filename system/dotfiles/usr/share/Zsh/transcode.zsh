# Functions for converting between different media formats.
#
# Requires GNU Parallel and FFmpeg. On macOS: brew install ffmpeg parallel

# Encode all *.flac files in the current directory as mp3.
flac2mp3() {
  parallel ffmpeg -i {} -qscale:a 0 {.}.mp3 ::: ./*.flac
}

# Encode all *.wma files in the current directory as mp3.
wma2mp3() {
  parallel ffmpeg -i {} -acodec libmp3lame -ab 192k {.}.mp3 ::: ./*.wma
}