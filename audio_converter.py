# imports for file & directory related operations etc.
import os
import glob
import shutil
import csv

import argparse # handles execution of batch operations from command line
import datetime

# imports for array manipulation & image processing
import numpy as np
from PIL import Image

import sys
# append paths to the locations of:
# 1. common_file_ops.py
# 2. pretty_midi_examples folder (cloned from the pretty_midi github)
sys.path.append("path/to/common_file_ops.py")
sys.path.append("path/to/pretty_midi_examples")

from common_file_ops import *
import pretty_midi
from pretty_midi_examples import reverse_pianoroll
import py_midicsv # package with functions used to convert between midi & csv

# Paths to executables used for conversions 
FFMPEG_PATH = 'path/to/ffmpeg/bin/ffmpeg.exe'
WAON_PATH = 'path/to/waon-0.11-mingw/waon.exe'
ARSS_PATH = "path/to/arss-0.2.3-windows/arss.exe"
TIMIDITY_PATH = "path/to/TiMidity++/timidity.exe"

# Functions for encoding/decoding integer values [used for transcribing piano roll note & velocity info to text files]
# For compatability with the functions performing the transcription, custom encoding/decoding functions should have a signature like hex2 & operate on two modes as seen in hex2
def hex2(n,direction):
    if direction == 'forward':
        return format(n,'02x').upper()
    elif direction == 'reverse':
        return int(n,16)


# ------------------------------------------------------------------------------------------------- #
## Functions to perform conversions on a single file:

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

timidity_options = ['-Ow','-o']

def midi_to_wav(source_path,dest_path,options_list=timidity_options):
    '''
    Converts a .mid file to a .wav file using Timidity++ (timidity.exe)
    Parameters:
        source_path = str/os.path; path to .mid file 
        dest_path = str/os.path; path to .wav file or to directory (default: use name of source .mid file for output .wav file)
        options_list = list; command line options passed to timidity: meant for the output file (default: ['-Ow','-o'])
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .wav file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".wav")

    # prep cmd list for subprocess.Popen()
    full_options = [source_path]+options_list+[dest_path]
    run_exec(TIMIDITY_PATH,full_options)

spec_analysis_options = ['--quiet','--analysis','-min','27.5','-max','19912.127','--bpo','12','--pps','25','--brightness','1']
wav_sine_synth_options = ['--quiet','--sine','-min','27.5','-max','19912.127','--pps','25','-r','44100','-f','16']
wav_noise_synth_options = ['--quiet','--noise','-min','27.5','-max','19912.127','--pps','25','-r','44100','-f','16']

def wav_to_spectro(source_path,dest_path,options_list=spec_analysis_options,encode=True):
    '''
    Converts a .wav file to a spectrogram (.png file) using ARSS - http://arss.sourceforge.net
    Parameters:
        source_path = str/os.path; path to .wav file 
        dest_path = str/os.path; path to spectrogram (.png file) or to directory (default: use name of source .wav file for output .png file)
        options_list = list; command line options passed to ARSS:
                    selects analysis mode, frequency range, beats per octave, pixels/sec, etc.
        encode = boolean; whether or not to re-encode the generated .png spectrogram as rgba - appends '-enc' to output file name
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path) # original splits
    
    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .png file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".png")

    arss_options = ['-i', source_path, '-o', dest_path] + options_list
    res = run_exec(ARSS_PATH,arss_options)
    if encode:
        # if required to re-encode the png, will attempt to do so using ffmpeg.

        dest_splits = path_splitter(dest_path) # take splits again in case the path was modified earlier 
        enc_path = os.path.join(dest_splits['directory'],dest_splits['name']+"-enc.png")
        try:
            img_fmt_converter(dest_path,enc_path,"rgba")
        except Exception:
            print("Unable to re-encode the .png file...")
        finally:
            os.remove(dest_path)

def spectro_to_wav(source_path,dest_path,options_list=wav_noise_synth_options):
    '''
    Converts a spectrogram (.png file) to a .wav file using ARSS - http://arss.sourceforge.net
    Parameters:
        source_path = str/os.path; path to spectrogram (.png file)
        dest_path = str/os.path; path to .wav file or to directory (default: use name of source .png file for output .wav file)
        options_list = list; command line options passed to ARSS:
                    selects synthesis mode (noise/sine), frequency range, beats per octave, pixels/sec, etc.
                    (can choose default module-level lists)
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

    arss_options = ['-i', inter_path, '-o', dest_path] + options_list
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

# ------------------------------------------------------------------------------------------------- #
## Functions to process large batches of files 

def midi_to_arss_batch(midi_folder,arss_folder,split_size=5,delete_wav=True):
    '''
    Takes a directory of .mid files and converts them to .wav files. 
    Then converts the .wav files into several ARSS spectrograms representing audio clips of `split_size` seconds.
    Parameters:
        midi_folder = str/os.path; path to root directory containing .mid files (will recursively search in directory for .mid files)
        arss_folder = str/os.path; path to directory where ARSS spectrogram slices will be stored
        split_size = float; duration of ARSS spectrogram slices in seconds
        delete_wav = boolean; whether or not to delete the intermediate .wav files
    '''
    # 1. create temporary directories under source_folder
    wav_folder = os.path.join(midi_folder,"WAVs from MIDIs")
    wav_splits = os.path.join(wav_folder,"splits")
    wav_archive = os.path.join(wav_folder,"raw_wav_archive")

    for f in [wav_folder,wav_splits,wav_archive]:
        ensure_dir(f)

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Starting from MIDI Files
    # 2. Convert all MIDI files in source path (recursively search within the directory) to WAV files
    midi_counter = 0
    midi_glob = glob.glob(midi_folder+"/**/*.mid", recursive=True)

    for filepath in midi_glob:
        src_folder,f_name = os.path.split(filepath)
        midi_to_wav(filepath,wav_folder)
        midi_counter += 1
    
    midi_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 2. split WAV files into chunks
    # archive full size WAV files after splitting
    for filepath in glob.glob(wav_folder+"/*.wav"):
        _, f_name = os.path.split(filepath)
        file_segmenter(filepath,wav_splits,split_size)
        shutil.move(filepath,os.path.join(wav_archive,f_name))
    
    # 3b. clean up WAV chunks -> delete raw WAV chunks
    for filepath in glob.glob(wav_splits+"/*.wav"):
        f_splits = path_splitter(filepath)
        cleanpath = os.path.join(wav_splits,f_splits['name']+"[cleaned metadata]"+f_splits['extension'])
        
        metadata_remover(filepath,cleanpath)
        os.remove(filepath)
    
    wav_chunk_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 4. finally convert clean WAV chunks to ARSS spectrograms
    arss_counter = 0
    for filepath in glob.glob(wav_splits+"/*.wav"):
        _, f_name = os.path.split(filepath)
        wav_to_spectro(filepath,arss_folder)
        arss_counter += 1

    # 5. Remove created directories used to save WAV files & splits (based on delete_wav)
    if delete_wav:
        for f in [wav_folder,wav_splits,wav_archive]:
            os.rmdir(f)

    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print('------------------------------------------------------------------------------------------')
    print("Summary:")
    print("Started process at: "+start_time)
    print("MIDI to WAV conversion completed at: "+midi_time)
    print("WAV chunking & cleaning completed at: "+wav_chunk_time)
    print("Spectrogram conversion completed at (end of process): "+end_time)
    print("Converted "+str(midi_counter)+" .mid files into...")
    print(str(arss_counter)+" "+str(split_size)+"-second sized spectrogram .png files!")
    print('------------------------------------------------------------------------------------------')
    

def midi_to_rollPics_batch(midi_folder,rolls_folder,split_size=5,conv_resolution=25,to_brighten=True,compress_colors=True):
    '''
    Takes a directory of .mid files and converts them to piano roll image slices (.png files). 
    Parameters:
        midi_folder = str/os.path; path to root directory containing .mid files (will recursively search in directory for .mid files)
        rolls_folder = str/os.path; path to directory where piano roll image slices will be stored
        split_size = float; duration of piano roll image slices in seconds
        conv_resolution = int; Sampling frequency for piano roll columns (each column is separated by 1/pps seconds)
        to_brighten = boolean; determine pixel brightnesses of output piano roll image slices: True=(0,255), False=(0,127)
        compress_colors = boolean; whether or not piano roll columns are compressed using the 3 color channels of the image
    '''
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # create suffixes for subfolder names and create the subfolders
    brightness_lvl = "(0-127)"
    if to_brighten:
        brightness_lvl = "(0-255)"
    f_prefix_full = "full, "
    f_prefix_seg = "t="+str(split_size)+", "
    f_suffix = "pps="+str(conv_resolution)+", b="+brightness_lvl
    if compress_colors:
        f_suffix +=" -cc"

    full_track_folder = os.path.join(rolls_folder,f_prefix_full+f_suffix)
    segment_folder = os.path.join(rolls_folder,f_prefix_seg+f_suffix)

    for p in [full_track_folder,segment_folder]:
        ensure_dir(p)
    
    # Starting from MIDI Files
    # 1. Convert all MIDI files in source path to midi rolls
    print("Creating piano rolls for full tracks:")
    midi_counter = 0
    failed_files,error_counter = [], 0
    midi_glob = glob.glob(midi_folder+"/**/*.mid", recursive=True)
    for filepath in midi_glob:
        conv = midi_to_roll(filepath,full_track_folder,pps=conv_resolution,brighten=to_brighten,compress_colors=compress_colors)
        if conv == 1:
            midi_counter += 1
        else:
            error_counter += 1
            failed_files.append(filepath)
        print("Rolls created: {}/{}".format(midi_counter,len(midi_glob)), end='\r')
    print()
    print("Failed to convert {} .mid files into .png rolls:".format(error_counter))
    for f in failed_files:
        print(f)
    midi_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    midi_counter = 0
    chunk_counter = 0
    print()
    print("Splitting full piano rolls into "+str(split_size)+"-sec sized chunks:")
    roll_glob = glob.glob(full_track_folder+"/*.png")
    print("number of rolls in "+full_track_folder+": "+str(len(roll_glob)))
    for filepath in roll_glob:
        _,f_name = os.path.split(filepath)
        chunk_counter += rollPic_slicer(filepath,segment_folder,pps=conv_resolution,compress_colors=compress_colors,slice_dur=split_size)
        midi_counter += 1
        print("Chunks created: {}".format(chunk_counter), end='\r')
    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print()
    print('------------------------------------------------------------------------------------------')
    print("Summary:")
    print("Started process at: "+start_time)
    print("MIDI to Roll conversion completed at: "+midi_time)
    print("Roll chunking completed at (end of process): "+end_time)
    print("Converted "+str(midi_counter)+" .mid files into...")
    print(str(chunk_counter)+" "+str(split_size)+"-second sized .png roll chunks!")
    print('------------------------------------------------------------------------------------------')

def midi_to_rollTxt_batch(midi_folder,txt_path,conv_resolution=25,note_enc_fn=hex2,velo_enc_fn=hex2):
    '''
    Takes a directory of .mid files, converts them to text which encodes piano roll notes & velocities.
    Appends the text of all .mid files together in one large .txt file.
    midi_folder = str/os.path; path to root directory containing .mid files (will recursively search in directory for .mid files)
    txt_path = str/os.path; path to .txt file which will store the text forms of the .mid files
    conv_resolution = int; number of text lines per second of audio (resolution)
    note_enc_fn, velo_enc_fn = functions used to encode piano roll note/velocity values from float to str
    '''
    ## bulk conversion of midi files to one large text file as encoded notes
    midi_counter = 0
    line_count = 0
    midi_glob = glob.glob(midi_folder+"/**/*.mid", recursive=True)
    for filepath in midi_glob:
        t = midi_to_rollTxt(filepath,txt_path,pps=conv_resolution,note_enc=note_enc_fn,velo_enc=velo_enc_fn)
        if t != 0:
            line_count += t
            midi_counter += 1
    print("Total midis: "+str(midi_counter))
    print("Total timesteps: "+str(line_count))
# ----------------------------------------------------------------------------------------- #
# Command line functionality for bulk conversion operations #
def batchOps_from_command_line():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Setting up choices & help text for bulk operations
    bco_choices = {
        'm_spec': (midi_to_arss_batch,'MIDI files -> ARSS spectrogram image chunks'),
        'm_rpic': (midi_to_rollPics_batch,'MIDI files -> Piano roll image chunks'),
        'm_rtxt': (midi_to_rollTxt_batch,'MIDI files -> Piano roll Encoded Text (one large .txt file output)\n [currently supported note & velocity encodings:\n 1) 2-digit hexadecimal - default]')
    }
    bco_help_msg = 'Choice of batch conversion operation:\n'
    for k,v in bco_choices.items():
        bco_help_msg += "{}: {}".format(k,v[1])+"\n"


    parser.add_argument('batch_conv_operation','-bop',type=str,choices=list(bco_choices.keys()),help=bco_help_msg)
    parser.add_argument('--src_dir','-i',type=str,help='Directory containing files to convert')
    parser.add_argument('--dest_dir','-o',type=str,help='Directory to save converted files')
    parser.add_argument('--chunk_size','-t',type=int,default=5,help= 'Duration for chunks (in secs) (default: %(default)s)')
    parser.add_argument('--conv_resolution','-r',type=int,default=25,help='<For piano rolls> Sampling frequency:\n time separation b/w columns = 1/resolution (default: %(default)s)')
    parser.add_argument('--spectro_keep_wav','-kw',action='store_true',help='<For ARSS Spectrograms> Keep intermediate wav files? (will delete by default if option omitted)')
    parser.add_argument('--brighten_rollPics','-b',action='store_true',help='<For piano roll pics> Brighten generated roll images?')
    parser.add_argument('--col_comp_rollPics','-cc',action='store_true',help='<For piano roll pics> Compress piano rolls using color dimension?')

    try:
        args = parser.parse_args()
        conv_fn = args.batch_conv_operation
        if conv_fn:
            print("Running batch conversion operation...")
            print("#####################################################################################")
            src,dest = args.src_dir,args.dest_dir
            if not src or not dest:
                raise ValueError("src_dir or dest_dir cannot be empty")
            if conv_fn == 'm_spec':
                del_wav = not args.spectro_keep_wav
                midi_to_arss_batch(src,dest,split_size=args.chunk_size,delete_wav=del_wav)
            elif conv_fn == 'm_rpic':
                midi_to_rollPics_batch(src,dest,split_size=args.chunk_size,conv_resolution=args.conv_resolution,to_brighten=args.brighten_rollPics,compress_colors=args.col_comp_rollPics)
            elif conv_fn == 'm_rtxt':
                midi_to_rollTxt_batch(src,dest,conv_resolution=args.conv_resolution)
            else:
                print("Invalid choice for batch_conv_operation...")
        
    except Exception as e:
        print("Exception occured, see traceback below:")
        print('-------------------------------------------------')
        print(e)

# attempt to run command line functionality
# if unable to do so, do nothing; this ensures that regular imports of the module will work.
try:
    batchOps_from_command_line()
except:
    pass
# ----------------------------------------------------------------------------------------- #

# # Other bulk operations; TODO: write separate functions for these and include with command line functionality # #
# use this space to quickly test these out