'''
Module containing functions used to perform common operations on files, file paths, etc.
'''
# imports of built-in packages
import subprocess
import os

# imports from package modules
## get paths to required executables from config.json
from .config import read_config
exec_paths = read_config()
FFMPEG_PATH = exec_paths["FFMPEG_PATH"]

# imports of external packages
import ffmpy

def path_splitter(path):
    directory,file_name = os.path.split(path)
    file_no_ext, file_ext = os.path.splitext(file_name)
    return {'directory': directory,'full_name': file_name, 'name': file_no_ext, 'extension': file_ext}

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def run_exec(exec_path,exec_options_list,input_data=None, stdout=None, stderr=None):
    cmd = [exec_path]+exec_options_list
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr
        )
    except OSError as e:
        print("OSError occured, perhaps executable path is invalid")
    output = process.communicate(input=input_data)
    if process.returncode != 0:
        print("Executable exited with process return code "+str(process.returncode))
    return output

def img_fmt_converter(source_path,dest_path,pix_format):
    ff = ffmpy.FFmpeg(
        executable=FFMPEG_PATH,
        global_options="-loglevel quiet",
        inputs= {source_path:None},
        outputs= {dest_path:'-pix_fmt '+pix_format}
    )
    ff.run()


def file_segmenter(source_path,dest_folder,segment_size_in_seconds,segment_suffix='_sp-%d'):
    # will split file at `source_path` into mulitple files which are stored in `dest_folder` and suffixed using the format string `segment_suffix`
    src_splits = path_splitter(source_path)
    dest_path = os.path.join(dest_folder,src_splits['name']+segment_suffix+src_splits['extension'])
    ff = ffmpy.FFmpeg(
        executable=FFMPEG_PATH,
        global_options="-loglevel quiet",
        inputs= {source_path:None},
        outputs= {dest_path:'-map 0 -c copy -f segment -segment_time '+str(segment_size_in_seconds)}
    )

    print(ff.cmd)
    ff.run()

# function to remove metadata from files (created to make WAV files usable with ARSS)
# will convert the file at `source_path` from its original format to a file with the format extension in `dest_path`
def metadata_remover(source_path,dest_path):
    # create filepath for temporary .wav file
    dest_splits = path_splitter(dest_path)
    temp_path = os.path.join(dest_splits['directory'],dest_splits['name']+"_temp"+dest_splits['extension'])

    # first convert to desired end format (with metadata)
    ff1 = ffmpy.FFmpeg(
        executable=FFMPEG_PATH,
        global_options="-loglevel quiet",
        inputs= {source_path:None},
        outputs= {temp_path:'-f '+dest_splits['extension'][1:]+' -map 0 -map_metadata -1 -fflags +bitexact -flags:v +bitexact -flags:a +bitexact'}
    )

    print(ff1.cmd)
    ff1.run()

    # convert temp file to final file w/o metadata -> ARSS is happy with this
    ff2 = ffmpy.FFmpeg(
        executable=FFMPEG_PATH,
        global_options="-loglevel quiet",
        inputs= {temp_path:None},
        outputs= {dest_path:'-f '+dest_splits['extension'][1:]+' -map 0 -map_metadata -1 -fflags +bitexact -flags:v +bitexact -flags:a +bitexact'}
    )
    print(ff2.cmd)
    ff2.run()

    os.remove(temp_path)