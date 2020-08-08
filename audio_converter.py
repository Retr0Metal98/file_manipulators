'''
Module with functions for converting audio files across several formats
'''
# imports of built-in packages
import os
import sys
import csv

# imports from package modules
from file_manipulators.common_file_ops import path_splitter, run_exec, img_fmt_converter
from file_manipulators.config import read_config

## get paths to required executables from config.json
exec_paths = read_config()
TIMIDITY_PATH = exec_paths["TIMIDITY_PATH"]
WAON_PATH = exec_paths["WAON_PATH"]
ARSS_PATH = exec_paths["ARSS_PATH"]
PRETTY_MIDI_EXAMPLES_PATH = exec_paths["PRETTY_MIDI_EXAMPLES_PATH"]
sys.path.append(PRETTY_MIDI_EXAMPLES_PATH)

# imports of external packages
## imports for array manipulation & image processing
import numpy as np
from PIL import Image
## imports for handling midi files
import pretty_midi
from pretty_midi_examples import reverse_pianoroll
import py_midicsv # package with functions used to convert between midi & csv


# Functions for encoding/decoding integer values [used for transcribing piano roll note & velocity info to text files]
# For compatability with the functions performing the transcription, custom encoding/decoding functions should have a signature like hex2 & operate on two modes as seen in hex2
def hex2(n,direction):
    '''
    Converts an integer to its 2-digit hexadecimal form (string) or vice versa
    Used for encoding/decoding piano roll note & velocity values to write/read text files
    Parameters:
        n = int; an integer
        direction = str; 'forward': int -> str, 'reverse': str -> int
    '''
    if direction == 'forward':
        return format(n,'02x').upper()
    elif direction == 'reverse':
        return int(n,16)

# ------------------------------------------------------------------------------------------------- #
## Functions to perform conversions on individual files

def wav_to_midi(source_path,dest_path):
    '''
    Converts a .wav file to .mid file using WaoN - http://waon.sourceforge.net
    Parameters:
        source_path = str/os.path; path to .wav file 
        dest_path = str/os.path; path to .mid file or to directory (default: use name of source .wav file for output .mid file)
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".mid")
    # prep cmd list for subprocess.Popen()
    waon_options = ['-i', source_path, '-o', dest_path]
    res = run_exec(WAON_PATH,waon_options)

timidity_options = '-Ow -o'

def midi_to_wav(source_path,dest_path,options=timidity_options):
    '''
    Converts a .mid file to a .wav file using Timidity++ (timidity.exe)
    Parameters:
        source_path = str/os.path; path to .mid file 
        dest_path = str/os.path; path to .wav file or to directory (default: use name of source .mid file for output .wav file)
        options = str; command line options passed to timidity: meant for the output file (default: '-Ow -o'])
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .wav file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".wav")

    # prep cmd list for subprocess.Popen()
    full_options = [source_path]+options.split()+[dest_path]
    run_exec(TIMIDITY_PATH,full_options)

spec_analysis_options = '--quiet --analysis -min 27.5 -max 19912.127 --bpo 12 --pps 25 --brightness 1'
wav_sine_synth_options = '--quiet --sine -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'
wav_noise_synth_options = '--quiet --noise -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'

def wav_to_spectro(source_path,dest_path,options=spec_analysis_options,encode=True):
    '''
    Converts a .wav file to a spectrogram (.png file) using ARSS - http://arss.sourceforge.net
    Parameters:
        source_path = str/os.path; path to .wav file 
        dest_path = str/os.path; path to spectrogram (.png file) or to directory (default: use name of source .wav file for output .png file)
        options = str; command line options passed to ARSS as a space separated string:
                    selects analysis mode, frequency range, beats per octave, pixels/sec, etc.
        encode = boolean; whether or not to re-encode the generated .png spectrogram as rgba - appends '-enc' to output file name
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path) # original path splits
    
    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .png file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".png")

    arss_options = ['-i', source_path, '-o', dest_path] + options.split()
    res = run_exec(ARSS_PATH,arss_options)
    if encode:
        # if required to re-encode the png, will attempt to do so using ffmpeg.

        dest_splits = path_splitter(dest_path) # take path splits again in case the path was modified earlier 
        enc_path = os.path.join(dest_splits['directory'],dest_splits['name']+"-enc.png")
        try:
            img_fmt_converter(dest_path,enc_path,"rgba")
        except Exception:
            print("Unable to re-encode the .png file...")
        finally:
            os.remove(dest_path)

def spectro_to_wav(source_path,dest_path,options=wav_noise_synth_options):
    '''
    Converts a spectrogram (.png file) to a .wav file using ARSS - http://arss.sourceforge.net
    Parameters:
        source_path = str/os.path; path to spectrogram (.png file)
        dest_path = str/os.path; path to .wav file or to directory (default: use name of source .png file for output .wav file)
        options = str; command line options passed to ARSS as a space separated string:
                    selects synthesis mode (noise/sine), frequency range, beats per octave, pixels/sec, etc.
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    # convert .png to bitmap internally
    inter_path = os.path.join(src_splits['directory'],src_splits['name']+"-24-bitmap.bmp")
    img_fmt_converter(source_path,inter_path,"bgr24")

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .wav file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".wav")

    arss_options = ['-i', inter_path, '-o', dest_path] + options.split
    res = run_exec(ARSS_PATH,arss_options)
    os.remove(inter_path)

def midi_to_roll(source_path,dest_path,pps=25,ret_array=False,brighten=True,compress_colors=True):
    '''
    Converts a .mid file to a piano roll using pretty_mid.get_piano_roll - https://craffel.github.io/pretty-midi/
    Parameters:
        source_path = str/os.path; path to .mid file
        dest_path = str/os.path; (for conversion to image) path to .png file or to directory (default: use name of source .mid file for output .png file)
        pps = int; Sampling frequency for piano roll columns (each column is separated by 1/pps seconds)
        ret_array = boolean; controls how the piano roll (raw 2D np.ndarray) is returned:
                    True = return as raw 2D np.ndarray to calling function [no need for dest_path] -> (roll_array)
                    False = process the raw 2D np.ndarray into an image file and write to dest_path -> (rollPic)
        brighten = boolean; if returning an image, whether or not to multiply pixel brightnesses by 2, i.e., bring them from the range (0,127) to (0,255)
        compress_colors = boolean; if returning an image, whether or not to compress the raw 2D np.ndarray into the 3 color channels of the output image:
                          True = 3 columns of the piano roll are represented by 1 column of pixels using the value of each column for the corresponding R,G,B channels
                          False = 3 columns of the piano roll are represented by 3 columns of pixels using the same value for the R,G,B channels
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .png file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".png")

    try:
        midi_container = pretty_midi.PrettyMIDI(source_path)
    except ValueError:
        # print("Unable to parse \""+source_path+"\" into .png roll")
        return -1
    roll_array = midi_container.get_piano_roll(fs=pps)
    if ret_array:
        return roll_array
    if brighten:
        roll_array *= 2 
    if compress_colors:
        if (roll_array.shape[1] % 3) != 0:
            pad_cols = 3 - (roll_array.shape[1] % 3)
            padding = np.zeros((roll_array.shape[0],pad_cols))
            roll_array = np.hstack((roll_array,padding))
        roll_array = roll_array.reshape((roll_array.shape[0],roll_array.shape[1]//3,3))
        roll_array = roll_array.astype(np.uint8)
        roll_array = np.ascontiguousarray(roll_array)
        
        roll_img = Image.fromarray(roll_array,mode='RGB').convert('RGB')
    else:
        roll_img = Image.fromarray(roll_array).convert('RGB')
    roll_img.save(dest_path)
    return 1

def rollPic_slicer(source_path,dest_folder,pps=25,compress_colors=True,slice_dur=5,slice_suffix="-sp{:03d}"):
    '''
    Cuts up a large piano roll image into slices (images) that represent a fixed duration of .mid file audio
    Parameters:
        source_path = str/os.path; path to large piano roll image
        dest_path = str/os.path; path to directory where roll image slices are to be stored
        pps = int; Sampling frequency used to generate the original roll image
        compress_colors = boolean; whether compression across color channels was performed to generate the original roll image
        slice_dur = int; max duration in seconds of each slice
        slice_suffix = str; formatting string used to name the slice images
    '''
    src_splits = path_splitter(source_path)

    roll_img = Image.open(source_path)
    roll_img.load()
    W,H = roll_img.size
    sl = slice_dur*pps
    if compress_colors:
        sl //= 3
    x = 0
    im_counter = 1
    while x < W:
        if x+sl > W:
            break
        dest_path = os.path.join(dest_folder,src_splits['name']+slice_suffix.format(im_counter)+".png")
        crop_area = (x,0,x+sl,H)
        chunk = roll_img.crop(crop_area)
        chunk.load()
        chunk.save(dest_path)
        im_counter += 1
        x += sl
    return im_counter

def roll_to_midi(source_path,dest_path,input_array=None,pps=25,compress_colors=True,limit_brightnesses=True): #using pretty_midi_examples/reverse_pianoroll.py
    '''
    Converts a piano roll (image/raw array) into a .mid file
    Parameters:
        source_path = str/os.path; (for conversion from image) path to piano roll image
        dest_path = str/os.path; path to .mid file or to directory (default: use name of source .png file appended with '-resynth' for output .mid file)
        input_array = array or None; if not None, will use as raw roll array instead of reading it from image at source_path
        pps = int; Sampling frequency used to generate the input roll image/array
        compress_colors = boolean; whether compression across color channels was performed to generate the input roll image/array
        limit_brightnesses = boolean; whether or not to limit brightnesses (pixel/array values) to the range (0,127) --needed for reverse_pianoroll.py to work
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+"-resynth.mid")
    
    roll_raw = []
    if input_array is not None:
        roll_raw = np.array(input_array,dtype='float32')
    else:
        roll_img = Image.open(source_path)
        roll_img.load()
        roll_raw = np.asarray(roll_img,dtype='float32')
    if compress_colors is True:
        roll_data = roll_raw.reshape((roll_raw.shape[0],int(roll_raw.shape[1]*3)))
    elif compress_colors is False:
        roll_data = np.mean(roll_raw,axis=2)
    elif compress_colors is None:
        roll_data = roll_raw

    if limit_brightnesses:
        for i in range(roll_data.shape[0]):
            for j in range(roll_data.shape[1]):
                if 128 <= roll_data[i][j] <= 255:
                    roll_data[i][j] //= 2
    roll_midi = reverse_pianoroll.piano_roll_to_pretty_midi(roll_data,fs=pps)
    roll_midi.write(dest_path)

def midi_to_csv(source_path,dest_path,ret_csv_strings=False):
    '''
    Converts a .mid file to a .csv file containing the playback instructions of the MIDI file using py-midicsv - https://pypi.org/project/py-midicsv/
    Parameters:
        source_path = str/os.path; path to .mid file
        dest_path = str/os.path; path to .csv file or to directory (default: use name of source .mid file for output .csv file)
        ret_csv_strings = boolean; whether to return list of csv formatted strings to calling function or write the list to a .csv file specified by dest_path
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .csv file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".csv")

    csv_str_list = py_midicsv.midi_to_csv(source_path)
    if ret_csv_strings:
        return csv_str_list

    with open(dest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_str_list)

# TODO: write a function for csv_to_midi

def midi_to_rollTxt(source_path,dest_path,pps=25,note_enc=hex2,velo_enc=hex2):
    '''
    Converts a .mid file to a .txt file containing an encoded form of its piano roll (notes+velocities)
    - Internally makes use of midi_to_roll function
    Parameters:
        source_path = str/os.path; path to .mid file
        dest_path = str/os.path; path to .txt file or to directory (default: use name of source .mid file for output .txt file)
        pps = int; Sampling frequency for piano roll columns (each column is separated by 1/pps seconds) -> passed to midi_to_roll
        for every second of audio: `pps` lines of encoded text to generated - 1 line = 1 piano roll column
        if no note is played in a column, a null value (note=0,velo=0) is written for that line
        note_enc, velo_enc = functions that specify how note/velocity values are to be encoded [see definition of hex2 function for an example]
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .txt file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".txt")
    
    print("Opened .mid file: ",source_path)
    # first get roll_array from midi
    roll_array = midi_to_roll(source_path,"",pps=pps,ret_array=True)
    try:
        roll_array = roll_array.T
    except Exception:
        print("failed to get roll_array")
        return 0
    roll_array = list(roll_array)
    num_notes, num_timesteps = len(roll_array[0]), len(roll_array)
    
    print("Number of notes:",num_notes)
    print("Number of timesteps:",num_timesteps)
    txt_out = open(dest_path,'a+')
    print("Written to .txt file at: "+dest_path)
    for t in range(num_timesteps):
        played_notes_encoded = ""
        for n in range(num_notes):
            velo = roll_array[t][n]
            if velo > 0:
                played_notes_encoded += note_enc(int(n),'forward')+velo_enc(int(velo),'forward')+" "
        
        played_notes_encoded = played_notes_encoded[:-1] # remove trailing space character
        if played_notes_encoded == "":
            played_notes_encoded = note_enc(int(0),'forward')+velo_enc(int(0),'forward')
        txt_out.write(played_notes_encoded) # TODO: add a 0000 note when no note is to be played for the line
        txt_out.write("\n")
    txt_out.write("\n")
    txt_out.close()
    return num_timesteps

def rollTxt_to_midi(source_path,dest_path,pps=25,note_enc=hex2,velo_enc=hex2,logging=True):
    '''
    Converts a .txt file containing an encoded form of a MIDI piano roll (notes+velocities) to a .mid file 
    - Internally makes use of roll_to_midi function
    Parameters:
        source_path = str/os.path; path to .txt file
        dest_path = str/os.path; path to .mid file or to directory (default: use name of source .txt file for output .mid file)
        pps = int; number of lines of encoded text to convert in order to produce 1 second of audio (sampling frequency) -> passed to roll_to_midi: 
        note_enc, velo_enc = functions that specify how note/velocity values are to be decoded [see definition of hex2 function for an example]
        logging = boolean; whether or not to print and write to a log file, statistics on the conversion process
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".mid")
        dest_splits = path_splitter(dest_path)
    
    try:
        txt_in = open(source_path,'r')
    except IOError:
        print("File at {} does not exist... Exiting program".format(source_path))
    
    raw_lines = txt_in.readlines()
    T_roll_rows = []
    note_len,velo_len = 0, 0
    transcribed_notes, dropped_notes, bad_notes, invalid_notes = 0, 0, 0, 0
    
    if note_enc == hex2 and velo_enc == hex2:
        note_len,velo_len = 2,2

    split_len = note_len+velo_len+1
    for line in raw_lines:
        splits = [line[i:i+split_len] for i in range(0,len(line),split_len)]
        roll_row = np.zeros((128,))
        for s in splits:
            if s == '\n':
                continue # no note on that line
            if len(s) != split_len:
                # print("Dropped note: ",s)
                dropped_notes += 1
                continue 
            else:
                # take the first 4 characters
                try:
                    note,velo = note_enc(s[:note_len], 'reverse'), float(velo_enc(s[note_len:split_len], 'reverse'))
                except ValueError:
                    # print("Bad note: ",s)
                    bad_notes += 1
                else:
                    if (0 <= note <= 127) and (0 <= velo <= 127):
                        transcribed_notes += 1
                        roll_row[note] = velo
                    else:
                        # print("Note/Velo value out of valid range: ",s)
                        invalid_notes += 1
                        continue
        T_roll_rows.append(roll_row)
    
    roll_array = np.vstack(T_roll_rows)
    roll_array = roll_array.T
    roll_to_midi(source_path,dest_path,input_array=roll_array,pps=pps,compress_colors=None,limit_brightnesses=False)
    if logging:
        log_strings = [
            '-------------------------------------------------',
            "Converted rollTxt {} to midi!".format(source_path),
            '*************************************************',
            "Transcribed {} notes".format(transcribed_notes),
            "Dropped {} incomplete notes".format(dropped_notes),
            "Ignored {} bad notes".format(bad_notes),
            "Ignored {} invalid notes".format(invalid_notes),
            '-------------------------------------------------',
        ]
        log_path = os.path.join(dest_splits['directory'],src_splits['name']+"-rollTxt_conv_log.txt")
        log_file = open(log_path,'a+')
        for line in log_strings:
            print(line)
        log_file.writelines(line+"\n" for line in log_strings)
        log_file.close()
