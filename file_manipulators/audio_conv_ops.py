'''
Module with functions for converting audio files across several formats
'''
# imports of built-in packages
import os
import sys
import csv
# imports from package modules
from .common_file_ops import path_splitter, run_exec, img_fmt_converter
from .config import read_config

## get paths to required executables from config.json
exec_paths = read_config()
TIMIDITY_PATH = exec_paths["TIMIDITY_PATH"]
WAON_PATH = exec_paths["WAON_PATH"]
ARSS_PATH = exec_paths["ARSS_PATH"]
PRETTY_MIDI_EXAMPLES_PATH = exec_paths["PRETTY_MIDI_EXAMPLES_PATH"]
sys.path.append(PRETTY_MIDI_EXAMPLES_PATH)

# imports of external packages
## imports for array manipulation, image processing, audio output
import numpy as np
from PIL import Image
import scipy.io.wavfile
## imports for handling midi files
import pretty_midi
from pretty_midi_examples import reverse_pianoroll,chiptunes
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

# Utility functions
def instrument_to_insCode(instrument):
    '''
    Given a pretty_midi Instrument object, returns a concise string code representing its program number & its is_drum parameter
    Parameters:
        instrument = pretty_midi.Instrument()
    Returns:
        ins_code = str; program code (as dec) + "d" if is_drum is True else "n"
    '''
    code = str(instrument.program)
    
    if instrument.is_drum:
        code += "d"
    else:
        code += "n"    
    return code


def insCode_to_instrument(insCode):
    '''
    Given the string code for an instrument, returns the pretty_midi Instrument object
    Parameters:
        insCode = str; string which encodes the program number and is_drum bool of the Instrument
    '''
    prog,drum = int(insCode[:-1]),insCode[-1:]

    name_str = 'ins_{}'.format(prog)
    is_drum = False
    if drum == 'd':
        is_drum = True
        name_str += '-drum'

    instrument = pretty_midi.Instrument(prog,is_drum=is_drum,name=name_str)
    return instrument


def drum_ins_to_roll(drum_ins,fs=25):
    '''
    Given a drum Instrument (pretty_midi.Instrument instance), return a rank-2 array: a drum roll
    Does not process pitch_bends and control_changes
    Parameters:
        drum_ins = pretty_midi.Instrument; 
            Instrument instance with is_drum attribute = True
        fs = int; 
            Sampling frequency for drum roll columns (each column is separated by 1/fs seconds)
    '''
    if drum_ins.notes == []:
        return np.array([[]]*128)
    end_time = drum_ins.get_end_time()
    drum_roll = np.zeros((128, int(fs*end_time)))
    # Add up drum roll matrix, note-by-note
    for note in drum_ins.notes:
        # Should interpolate
        drum_roll[note.pitch,int(note.start*fs):int(note.end*fs)] += note.velocity
    return drum_roll
    # converting drum_roll back to instrument is covered in rollArr3R_to_PrettyMIDI


def rollArr2R_to_Img(roll_array,brighten,compress_colors):
    '''
    Converts a rank-2 piano_roll_array to a PIL Image object
    Parameters:
        roll_array = rank-2 np.ndarray; 
            piano roll of 1 instrument
        brighten = boolean; 
            whether or not to multiply pixel brightnesses by 2, i.e., bring them from the range (0,127) to (0,255)
        compress_colors = boolean; 
            whether or not to compress the raw 2D np.ndarray into the 3 color channels of the output image:
                True = 3 columns of the piano roll are represented by 1 column of pixels using the value of each column for the corresponding R,G,B channels
                False = 3 columns of the piano roll are represented by 3 columns of pixels using the same value for the R,G,B channels
    '''
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
    return roll_img


def rollArr3R_to_PrettyMIDI(roll_array,ins_codes,fs=25):
    '''
    Converts a rank-3 piano_roll_array to a PrettyMIDI class instance using instrument codes 
    Expands on the function piano_roll_to_pretty_midi from https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py

    Parameters:
        roll_array = rank-3 np.ndarray; 
            must be of shape - (#notes,#timesteps,#instruments) filled with velocity values in the range (0,127)
        ins_codes = tuple/sequence; 
            MIDI program codes for the instruments used to generate the .mid file
        fs = int; 
            Sampling frequency for piano roll columns (each column is separated by 1/fs seconds)
    '''
    notes,frames,num_ins = roll_array.shape
    pm = pretty_midi.PrettyMIDI()
    instruments_used = [insCode_to_instrument(c) for c in ins_codes] # list of Instrument objects

    # pad 1 column of zeros for every instrument so we can acknowledge inital and ending events
    roll_array = np.pad(roll_array, [(0, 0), (1, 1), (0, 0)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(roll_array,axis=1))

    # keep track on velocities and note on times for each instrument
    prev_velocities = [np.zeros(notes, dtype=int) for i in range(num_ins)]
    note_on_time = [np.zeros(notes) for i in range(num_ins)]

    for note,time,ins in zip(*velocity_changes):
        # print("Ins: {}, Note: {}, Timestep: {}".format(ins,note,time))
        velocity = roll_array[note,time+1,ins]
        time = time / fs
        if velocity > 0:
            if prev_velocities[ins][note] == 0:
                note_on_time[ins][note] = time
                prev_velocities[ins][note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[ins][note],
                pitch=note,
                start=note_on_time[ins][note],
                end=time)
            instruments_used[ins].notes.append(pm_note)
            prev_velocities[ins][note] = 0
    
    pm.instruments = instruments_used
    return pm

# ------------------------------------------------------------------------------------------------- #
## Functions to perform conversions on individual files

def wav_to_midi(source_path,dest_path):
    '''
    Converts a .wav file to .mid file using WaoN - http://waon.sourceforge.net
    Parameters:
        source_path = str/os.path; 
            path to .wav file 
        dest_path = str/os.path; 
            path to .mid file or to directory (default: use name of source .wav file for output .mid file)
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
        source_path = str/os.path; 
            path to .mid file 
        dest_path = str/os.path; 
            path to .wav file or to directory (default: use name of source .mid file for output .wav file)
        options = str; 
            command line options passed to timidity: meant for the output file (default: '-Ow -o'])
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


def midi_to_wav_prettyMIDI(source_path,dest_path,fs=44100,drum_vol_reduction=4):
    '''
    Converts a .mid file to a .wav file using the synthesize functions in the pretty_midi package
    Drum tracks are synthesized using the function pretty_midi/examples/chiptunes.py
    Parameters:    
        source_path = str/os.path; 
            path to .mid file 
        dest_path = str/os.path; 
            path to .wav file or to directory (default: use name of source .mid file for output .wav file)
        fs = int; 
            Sample rate for .wav file
        drum_vol_reduction = int; 
            factor by which to divide the amplitudes of the drum track waveforms
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .wav file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".wav")
    midi_container = pretty_midi.PrettyMIDI(source_path)
    waveforms = []
    for ins in midi_container.instruments:
        if ins.is_drum:
            wave_f = chiptunes.synthesize_drum_instrument(ins,fs=fs)
            wave_f /= drum_vol_reduction
        else:
            wave_f = ins.synthesize(fs=fs)
        waveforms.append(wave_f)
    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform
    # Normalize
    synthesized /= np.abs(synthesized).max()
    synthesized = synthesized.astype(np.float32)
    # Finally write to .wav file
    scipy.io.wavfile.write(dest_path,fs,synthesized)


def midi_to_roll(source_path,dest_path,fs=25,out_type='one_roll',brighten=True,compress_colors=True):
    '''
    Converts a .mid file to piano roll(s) using get_piano_roll on each instrument - https://craffel.github.io/pretty-midi/
    Parameters:
        source_path = str/os.path; 
            path to .mid file
        dest_path = str/os.path; 
            for conversion to images: path to .png file [suffixed with instrument code] or to directory (default: use name of source .mid file as prefix for output .png files)
        fs = int; 
            Sampling frequency for piano roll columns (each column is separated by 1/fs seconds)
        out_type = str; 
            controls how the piano roll(s) is returned:
                'array_r3' = return as 1 raw rank-3 np.ndarray (instrument piano rolls merged into one) along with a tuple containing instrument codes to calling function [no need for dest_path]
                'array_r2' = return as 1 raw rank-2 np.ndarray (full .mid file transcripted with only instrument with prog number 0)
                'sep_roll' = process raw rank-2 np.ndarrays for each instrument into separate image files and write to dest_path 
                'one_roll' = default; convert the ENTIRE .mid file to one rank-2 piano roll array (can be read as one instrument) and write to one image file to dest_path
        brighten = boolean; 
            used when returning images, see rollArr2R_to_Img for details
        compress_colors = boolean; 
            used when returning images, see rollArr2R_to_Img for details
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
    ins_codes = []
    ins_rolls = []
    for ins in midi_container.instruments:
        code = instrument_to_insCode(ins)
        roll = []
        if ins.is_drum:
            roll = drum_ins_to_roll(ins,fs=fs)
        else:
            roll = ins.get_piano_roll(fs=fs)
        ins_codes.append(code)
        ins_rolls.append(roll)
    one_ins_roll = midi_container.get_piano_roll(fs=fs)

    if out_type == 'array_r3':
        max_len = max([r.shape[1] for r in ins_rolls])
        padded_rolls = []
        for i in range(len(ins_rolls)):
            r = ins_rolls[i]
            padded_r = np.pad(r,[(0,0),(0,max_len-r.shape[1])])
            padded_rolls.append(padded_r)
        stacked_roll_array = np.stack(padded_rolls,axis=-1)
        return (stacked_roll_array,ins_codes)
    elif out_type == 'sep_roll':
        dest_splits = path_splitter(dest_path)
        for i in range(len(ins_rolls)):
            roll_array,roll_code = ins_rolls[i],ins_codes[i]            
            save_path = os.path.join(dest_splits['directory'],dest_splits['name']+"-ins_"+roll_code+".png")
            roll_img = rollArr2R_to_Img(roll_array,brighten=brighten,compress_colors=compress_colors)
            roll_img.save(save_path)
        return len(ins_rolls)
    else:
        # if not creating separate rolls for instruments, create a single roll for entire midi track with instrument prog=0 (Piano)
        if out_type == 'one_roll':
            roll_img = rollArr2R_to_Img(one_ins_roll,brighten=brighten,compress_colors=compress_colors)
            roll_img.save(dest_path)
            return 1
        elif out_type == 'array_r2':
            return one_ins_roll
        

def rollPic_slicer(source_path,dest_folder,fs=25,compress_colors=True,slice_dur=5,slice_suffix="-sp{:03d}"):
    '''
    Cuts up a large piano roll image into slices (images) that represent a fixed duration of .mid file audio
    Parameters:
        source_path = str/os.path; 
            path to large piano roll image
        dest_path = str/os.path; 
            path to directory where roll image slices are to be stored
        fs = int; 
            Sampling frequency used to generate the original roll image
        compress_colors = boolean; 
            whether compression across color channels was performed to generate the original roll image
        slice_dur = int; 
            max duration in seconds of each slice
        slice_suffix = str; 
            formatting string used to name the slice image files
    '''
    src_splits = path_splitter(source_path)

    roll_img = Image.open(source_path)
    roll_img.load()
    W,H = roll_img.size
    sl = slice_dur*fs
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


def roll_to_midi(source_path,dest_path,input_array=None,ins_codes=('0n',),fs=25,compress_colors=True,limit_rollvals=True): #using pretty_midi_examples/reverse_pianoroll.py
    '''
    Converts a piano roll (image/raw array) into a .mid file
    Parameters:
        source_path = str/os.path; 
            for conversion from image: path to piano roll image
        dest_path = str/os.path; 
            path to .mid file or to directory (default: use name of source .png file appended with '-resynth' for output .mid file)
        input_array = array or None; 
            if not None, will use as raw roll array instead of reading it from image at source_path
        ins_codes = tuple/sequence; 
            for input_array: String codes for the instruments  used to generate the .mid file (see instrument_to_insCode for details)
        fs = int; 
            Sampling frequency used to generate the input roll image/array
        compress_colors = boolean; 
            for conversion from image: whether compression across color channels was performed to generate the input roll image
        limit_rollvals = boolean; 
            whether or not to limit pixel/array values to the range (0,127) --needed for successful transcription
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+"-resynth.mid")
    
    roll_raw, roll_to_conv = [],[]
    if input_array is not None:
        roll_raw = np.array(input_array,dtype='float32')
        compress_colors=None
    else:
        # piano roll image is for only one instrument so it is first converted to rank-2
        roll_img = Image.open(source_path)
        roll_img.load()
        roll_raw = np.asarray(roll_img,dtype='float32')
    if compress_colors is True:
        roll_to_conv = roll_raw.reshape((roll_raw.shape[0],int(roll_raw.shape[1]*3)))
    elif compress_colors is False:
        roll_to_conv = np.mean(roll_raw,axis=2)
    elif compress_colors is None:
        roll_to_conv = roll_raw

    # for the conversion back to MIDI, roll_to_conv must be of rank-3
    if len(roll_to_conv.shape) < 3:
        roll_to_conv = roll_to_conv.reshape(roll_to_conv.shape+(1,))

    def lim_val(val):
        if 128 <= val <= 255:
            val //= 2
        return val
    if limit_rollvals:
        roll_to_conv = np.vectorize(lim_val)(roll_to_conv)

    roll_midi = rollArr3R_to_PrettyMIDI(roll_to_conv, ins_codes, fs=fs)
    roll_midi.write(dest_path)


def midi_to_csv(source_path,dest_path,ret_csv_strings=False):
    '''
    Converts a .mid file to a .csv file containing the playback instructions of the MIDI file using py-midicsv - https://pypi.org/project/py-midicsv/
    Parameters:
        source_path = str/os.path; 
            path to .mid file
        dest_path = str/os.path; 
            path to .csv file or to directory (default: use name of source .mid file for output .csv file)
        ret_csv_strings = boolean; 
            whether to return list of csv formatted strings to calling function or write the list to a .csv file specified by dest_path
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

def midi_to_rollTxt(source_path,dest_path,all_ins=False,fs=25,enc_fn=hex2):
    '''
    Converts a .mid file to a .txt file containing an encoded form of its piano roll (notes+velocities)
    - Internally makes use of midi_to_roll function
    Parameters:
        source_path = str/os.path; 
            path to .mid file
        dest_path = str/os.path; 
            path to .txt file or to directory (default: use name of source .mid file for output .txt file)
        all_ins = boolean; 
            decides whether to transcribe .mid file events including instrument information specified in the .mid file 
            or transcribe all events such that they can be read back with only 1 common instrument
        fs = int; 
            Sampling frequency for piano roll columns (each column is separated by 1/fs seconds) -> passed to midi_to_roll
            for every second of audio: `fs` lines of encoded text to generated - 1 line = 1 piano roll column
            if no note is played in a column, a null value (note=0,velo=0) is written for that line
        enc_fn = function;
            specifies how notes, velocities, and instrument codes are to be encoded [see definition of hex2 function for an example]
    '''
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .txt file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".txt")
    
    print("Opened .mid file: ",source_path)
    txt_out = open(dest_path,'a+')
    print("Writing to .txt file at: "+dest_path)
    # first get roll_array from midi 
    try:   
        if not all_ins:
            ## single instrument mode
            # roll to be transcribed as Acoustic Grand Piano - instrument program 0 (default)
            roll_array = midi_to_roll(source_path,"",fs=fs,out_type='array_r2')
            ins_codes = ("0n",) 
        else:
            #if transcribing with all instruments, each line represents a roll column and would have the following template:
            # <Instrument_0_code>: <Note+Velo> <Note+Velo> ...<TAB><Instrument_1_code>: <Note+Velo> <Note+Velo> ...<TAB><Instrument_2_code>: <Note+Velo> <Note+Velo> ...<TAB><NEWLINE>
                roll_array,ins_codes = midi_to_roll(source_path,"",fs=fs,out_type='array_r3')
    except Exception:
        print("failed to get roll_array")
        return 0
        # roll_array = list(roll_array)
        # num_notes, num_timesteps = len(roll_array[0]), len(roll_array)
        
        # print("Number of notes:",num_notes)
        # print("Number of timesteps:",num_timesteps)
        # 
        # for t in range(num_timesteps):
        #     played_notes_encoded = ""
        #     for n in range(num_notes):
        #         velo = roll_array[t][n]
        #         if velo > 0:
        #             played_notes_encoded += enc_fn(int(n),'forward')+enc_fn(int(velo),'forward')+" "
            
        #     played_notes_encoded = played_notes_encoded[:-1] # remove trailing space character
        #     if played_notes_encoded == "":
        #         played_notes_encoded = enc_fn(int(0),'forward')+enc_fn(int(0),'forward')
        #     txt_out.write(played_notes_encoded) 
        #     txt_out.write("\n")
        # txt_out.write("\n")
        # txt_out.close()
        # return num_timesteps
    
    encoded_ins_codes = [enc_fn(int(c[:-1]),"forward")+c[-1] for c in ins_codes] #ins_code is of the form: <number from 0-127><n/d based on is_drum>
    print(roll_array.shape)
    col_count = 0
    for c in range(roll_array.shape[1]):
        col = roll_array[:,c,:]
        col_str = ""
        col_count += 1
        for i in range(roll_array.shape[2]):
            ins_str = encoded_ins_codes[i]+"- "
            col_ins = col[:,i]
            non_zero_notes = list(np.nonzero(col_ins)[0]) #indices of notes with non-zero velocity values
            if len(non_zero_notes) == 0:
                # if no notes for instrument, add a null note
                ins_str += enc_fn(int(0),'forward')+enc_fn(int(0),'forward')
            else:
                # otherwise, add note-velo strings separated by spaces
                for n in non_zero_notes:
                    ins_str += enc_fn(int(n),'forward')+'-'+enc_fn(int(col_ins[n]),'forward')+" "
            ins_str += '\t'
            col_str += ins_str
        col_str += '\n'
        txt_out.write(col_str)
    txt_out.write("\n")
    txt_out.close()
    return col_count            


def rollTxt_to_midi(source_path,dest_path,fs=25,enc_fn=hex2,logging=True):
    '''
    Converts a .txt file containing an encoded form of a MIDI piano roll (notes+velocities) to a .mid file 
    - Internally makes use of roll_to_midi function
    Parameters:
        source_path = str/os.path; 
            path to .txt file
        dest_path = str/os.path; 
            path to .mid file or to directory (default: use name of source .txt file for output .mid file)
        fs = int; 
            number of lines of encoded text to convert in order to produce 1 second of audio (sampling frequency) -> passed to roll_to_midi: 
        enc = function; 
            specifies how note/velocity values are to be decoded [see definition of hex2 function for an example]
        logging = boolean; 
            whether or not to print and write to a log file, statistics on the conversion process
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
    
    if enc_fn == hex2:
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
                    note,velo = enc_fn(s[:note_len], 'reverse'), float(enc_fn(s[note_len:split_len], 'reverse'))
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
    roll_to_midi(source_path,dest_path,input_array=roll_array,fs=fs,limit_rollvals=False)
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


spec_analysis_options = '--quiet --analysis -min 27.5 -max 19912.127 --bpo 12 --pps 25 --brightness 1'
wav_sine_synth_options = '--quiet --sine -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'
wav_noise_synth_options = '--quiet --noise -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'

def wav_to_spectro(source_path,dest_path,options=spec_analysis_options,encode=True):
    '''
    Converts a .wav file to a spectrogram (.png file) using ARSS - http://arss.sourceforge.net
    Parameters:
        source_path = str/os.path; 
            path to .wav file 
        dest_path = str/os.path; 
            path to spectrogram (.png file) or to directory (default: use name of source .wav file for output .png file)
        options = str; 
            command line options passed to ARSS as a space separated string:
            selects analysis mode, frequency range, beats per octave, pixels/sec, etc.
        encode = boolean; 
            whether or not to re-encode the generated .png spectrogram as rgba - appends '-enc' to output file name
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
        source_path = str/os.path; 
            path to spectrogram (.png file)
        dest_path = str/os.path; 
            path to .wav file or to directory (default: use name of source .png file for output .wav file)
        options = str; 
            command line options passed to ARSS as a space separated string:
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