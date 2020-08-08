# file_manipulators
Repository of programs used to manipulate files - convert across file types, collate/segment them, etc.

## Dependencies
- Executables: 
  - FFmpeg: https://ffmpeg.org/download.html
  - Timidity++ - an open source MIDI to WAVE converter and player: https://sourceforge.net/projects/timidity/
  - WaoN - a wave-to-notes transcriber: http://waon.sourceforge.net
  - The Analysis & Resynthesis Sound Spectrograph (ARSS): http://arss.sourceforge.net
- Python packages:
  - In built packages: `sys`, `os`, `subprocess`, `glob`, `shutil`, `csv`, `argparse`, `datetime`
  - `numpy`: https://pypi.org/project/numpy/
  - `pillow`/`PIL`: https://pypi.org/project/Pillow/
  - `matplotlib`: https://pypi.org/project/matplotlib/
  - `ffmpy`: https://pypi.org/project/ffmpy/
  - `pretty_midi`: https://pypi.org/project/pretty_midi/ https://craffel.github.io/pretty-midi/
    - Also needs a clone of the `pretty_midi_examples` folder from `pretty_midi`'s github repo
  - `py_midicsv`: https://pypi.org/project/py-midicsv/

## Setup
- Download & install all required executables and python packages
- Clone/Download & unzip this repo to a local directory
- From within the `file_manipulators` folder, run `python config.py write` and enter the paths of the installed executables & packages as prompted

## Usage
- Before importing this package to your own programs, include the following:
  - `import sys`
  - `sys.path.append('path/to/parent/directory/of/file_manipulators/folder')`
  
