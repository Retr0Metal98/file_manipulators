# file_manipulators
Repository of programs used to manipulate files - convert across file types, collate/segment, etc.

## common_file_ops.py
A file containing functions to perform common operations on files, filepaths, etc.
- Dependencies:
  - Python packages: subprocess, os, ffmpy
  - `FFmpeg`: download the executables from https://ffmpeg.org/download.html
- Setup:
  - Change the global variable `FFMPEG_PATH` to a string containing the local path to `ffmpeg.exe`

## audio_converter.py
