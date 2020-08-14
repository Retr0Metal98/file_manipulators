'''
Module with functions for converting audio files across several formats
'''
# imports of built-in packages
import os
import sys
import csv
import re

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

# ------------------------------------------------------------------------------------------------- #
# Functions for encoding/decoding integer values [used for transcribing piano roll note & velocity info to text files]
# For compatability with the functions performing the transcription, custom encoding/decoding functions should have a signature like hex2 & operate on two modes as seen in hex2

def hex2(n,direction):
    """Converts an integer to its 2-digit hexadecimal form (string) or vice versa.

    Parameters
    ----------
    n : int/str
        a positive integer or 2-digit hexadecimal string.
    direction : str
        direction of conversion: int to str, 'reverse': str to int.

    Returns
    -------
    out : str/int
        hexadecimal string of input int (or) base-10 int of input hexadecimal string.
    """    
    if direction == 'forward':
        out = format(n,'02x').upper()
    elif direction == 'reverse':
        out = int(n,16)
    return out


# ------------------------------------------------------------------------------------------------- #
# Utility functions

def instrument_to_insCode(instrument):
    """Given a pretty_midi.Instrument object, returns a concise string code representing its program number & its is_drum parameter.

    Parameters
    ----------
    instrument : pretty_midi.Instrument
        object of pretty_midi.Instrument class.

    Returns
    -------
    code : str 
        concatenation of MIDI program code (as decimal) and ("d" if instrument.is_drum is True else "n").
    """    
    code = str(instrument.program)
    
    if instrument.is_drum:
        code += "d"
    else:
        code += "n"    
    return code


def check_insCode(insCode,print_res=False):
    """Checks if a given instrument code string is valid, optionally prints the result of the check.

    Parameters
    ----------
    insCode : str
        string to check if it represents a valid instrument code.
    print_res : bool, optional
        prints the result of checking the input string, by default False.

    Returns
    -------
    res : bool
        whether or not the input insCode was valid.
    """    
    prog,drum = int(insCode[:-1]),insCode[-1:]
    insName = ''
    if drum == 'n':
        insName = pretty_midi.program_to_instrument_name(prog)
    elif drum == 'd':
        insName = pretty_midi.note_number_to_drum_name(prog+35)
    # last character must be 'd' or 'n'

    res_msg = "insCode is invalid and doesn't refer to any instrument."    
    res = False
    if insName:
        res = True
        res_msg = "insCode {} refers to the instrument {}.".format(insCode,insName)
    if print_res:
        print(res_msg)
    return res


def insCode_to_instrument(insCode):
    """Given the string code for an instrument, returns the corresponding pretty_midi.Instrument object.

    Parameters
    ----------
    insCode : str
        string which encodes the program number and is_drum bool of the Instrument.

    Returns
    -------
    instrument : pretty_midi.Instrument
        pretty_midi.Instrument object corresponding to the input insCode.
    """
    prog,drum = int(insCode[:-1]),insCode[-1:]

    name_str = 'ins_{}'.format(prog)
    is_drum = False
    if drum == 'd':
        is_drum = True
        name_str += '-drum'

    instrument = pretty_midi.Instrument(prog,is_drum=is_drum,name=name_str)
    return instrument


def drum_ins_to_roll(drum_ins,fs=25):
    """Given a drum Instrument (pretty_midi.Instrument object), return a 2D array: a drum roll containing velocities of each note for all timesteps. Does not process pitch bends and control changes.

    Parameters
    ----------
    drum_ins : pretty_midi.Instrument
        Instrument object with is_drum attribute set to True (represents MIDI drum instrument).
    fs : int, optional
        Sampling frequency for drum roll columns (each column is separated by 1/fs seconds), by default 25.

    Returns
    -------
    drum_roll : np.ndarray, shape=(128,timesteps)
        2D piano roll of MIDI data for the input drum instrument.
    """      
    if drum_ins.notes == []:
        return np.array([[]]*128)
    end_time = drum_ins.get_end_time()
    drum_roll = np.zeros((128, int(fs*end_time)))
    # Add up drum roll matrix, note-by-note
    for note in drum_ins.notes:
        # Should interpolate
        drum_roll[note.pitch,int(note.start*fs):int(note.end*fs)] += note.velocity
    return drum_roll
    # converting drum_roll back to instrument is covered in rollArr3D_to_PrettyMIDI


def rollArr2D_to_Img(roll_array,brighten,compress_colors):
    """Converts a 2D piano roll of MIDI data to a PIL Image object.

    Parameters
    ----------
    roll_array : np.ndarray, shape=(notes,timesteps)
        2D Piano roll of MIDI data.
    brighten : bool
        whether or not to multiply pixel brightnesses by 2, i.e., bring them from the range (0,127) to (0,255).
    compress_colors : bool
        whether or not to compress the raw 2D np.ndarray into the 3 color channels of the output image:
            True = 3 columns of the piano roll are represented by 1 column of pixels using the value of each column for the corresponding R,G,B channels.
            False = 3 columns of the piano roll are represented by 3 columns of pixels using the same value for the R,G,B channels.

    Returns
    -------
    roll_img : PIL.Image
        An Image object of the input roll_array converted to RGB format.
    """    
    if brighten:
        roll_array *= 2 
    if compress_colors:
        # pad the roll_array with empty columns (timesteps).
        # the resulting columns can then be evenly divided into groups of 3, allowing them to fit in the R,G,B channels of the output image
        if (roll_array.shape[1] % 3) != 0:
            pad_cols = 3 - (roll_array.shape[1] % 3)
            padding = np.zeros((roll_array.shape[0],pad_cols))
            roll_array = np.hstack((roll_array,padding)) 
        roll_array = roll_array.reshape((roll_array.shape[0],roll_array.shape[1]//3,3)) # reshape into 3D array
        roll_array = roll_array.astype(np.uint8)
        roll_array = np.ascontiguousarray(roll_array)
        
        roll_img = Image.fromarray(roll_array,mode='RGB').convert('RGB')
    else:
        roll_img = Image.fromarray(roll_array).convert('RGB')
    return roll_img


def rollArr3D_to_PrettyMIDI(roll_array,ins_codes,fs=25):
    """Converts a 3D piano roll of MIDI data (combination of 2D piano rolls of multiple instruments) to a PrettyMIDI object using input instrument codes.
    Expands on the function `piano_roll_to_pretty_midi` from https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py 

    Parameters
    ----------
    roll_array : np.ndarray, shape=(notes,timesteps,instruments)
        3D piano roll of MIDI data (2D piano rolls of multiple instruments padded to same shape and stacked along 'timesteps' axis).
    ins_codes : tuple/list
        MIDI program codes for the instruments to be used to generate the .mid file. See the function instrument_to_insCode for details.
        Length of ins_codes must match roll_array.shape[2].
    fs : int, optional
        Sampling frequency for piano roll columns (each column is separated by 1/fs seconds), by default 25.

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        PrettyMIDI object with all instruments and corresponding notes from the input roll_array.
    """    
    notes,frames,num_ins = roll_array.shape
    pm = pretty_midi.PrettyMIDI()
    instruments_used = [insCode_to_instrument(c) for c in ins_codes] # create list of Instrument objects from ins_codes

    # pad 1 column of zeros for every instrument so we can acknowledge inital and ending events
    roll_array = np.pad(roll_array, [(0, 0), (1, 1), (0, 0)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(roll_array,axis=1))

    # keep track on velocities and note on times for each instrument
    prev_velocities = [np.zeros(notes, dtype=int) for i in range(num_ins)]
    note_on_time = [np.zeros(notes) for i in range(num_ins)]

    for note,time,ins in zip(*velocity_changes):
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
    """Converts a .wav file to .mid file using WaoN - http://waon.sourceforge.net

    Parameters
    ----------
    source_path : 
        path to input .wav file.
    dest_path : str/os.path
        path to output .mid file or to directory (if directory: use name of source .wav file for output .mid file).
    """    
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
    """Converts a .mid file to a .wav file using Timidity++ (timidity.exe) - https://sourceforge.net/projects/timidity/

    Parameters
    ----------
    source_path : str/os.path
        path to input .mid file.
    dest_path : str/os.path
        path to .wav file or to directory (if directory: use name of source .mid file for output .wav file).
    options : str, optional
        space-separated string of command line options passed to timidity: meant for the output file, by default '-Ow -o'.
    """    
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
    """Converts a .mid file to a .wav file using the synthesize functions in the pretty_midi package.  
    Drum tracks are synthesized using the function `synthesize_drum_instrument` in pretty_midi/examples/chiptunes.py

    Parameters
    ----------
    source_path : str/os.path
        path to input .mid file.
    dest_path : str/os.path
        path to output .wav file or to directory (if directory: use name of source .mid file for output .wav file).
    fs : int, optional
        Sample rate for .wav file, by default 44100.
    drum_vol_reduction : int, optional
        factor by which to divide the amplitudes of the drum track waveforms, by default 4.
    """    
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '': 
        # if dest_path points to a directory and not a .wav file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".wav")
    
    midi_container = pretty_midi.PrettyMIDI(source_path)
    waveforms = []
    # get the waveforms of all instruments in the midi file
    for ins in midi_container.instruments:
        if ins.is_drum:
            # if instrument is a drum, get its waveform from chiptunes
            wave_f = chiptunes.synthesize_drum_instrument(ins,fs=fs)
            # reduce the amplitude of its waveform by drum_vol_reduction
            wave_f /= drum_vol_reduction
        else:
            # otherwise, get its waveform by calling its synthesize function (which will return an empty waveform if it is a drum instrument)
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


def midi_to_roll(source_path,dest_path,out_type,fs=25,brighten=True,compress_colors=True):
    """Converts a .mid file to piano roll(s) using pretty_midi.get_piano_roll (https://craffel.github.io/pretty-midi/) and/or drum_ins_to_roll functions.
    Piano roll(s) are then saved to image files or returned as arrays.

    Parameters
    ----------
    source_path : str/os.path
        path to input .mid file.
    dest_path : str/os.path
        used when returning images: path to .png file [suffixed with instrument code] or to directory (if directory: use name of source .mid file as prefix for output .png files).
    out_type : str
        controls how the piano roll(s) is returned, one of {'array_r3', 'array_r2', 'sep_roll, 'one_roll'}:
            
        'array_r3' = return as 1 raw 3D np.ndarray (instrument piano rolls merged into one) along with a tuple containing instrument codes to calling function [dest_path unused].

        'array_r2' = return as 1 raw 2D np.ndarray made by using pretty_midi.get_piano_roll on the .mid file [dest_path unused].
        (transcript all tracks in .mid file (except drum tracks) using default instrument: Acoustic Grand Piano -insCode="0n").

        'sep_roll' = process raw 2D np.ndarrays for each instrument into separate image files using rollArr2D_to_Img and write them to dest_path.

        'one_roll' = process the raw 2D np.ndarray from 'array_r2' mode into one image file using rollArr2D_to_Img and write it to dest_path.
    fs : int, optional
        Sampling frequency: number of columns in piano roll/image per second of audio (each column is separated by 1/fs seconds), by default 25.
    brighten : bool, optional
        used when returning images: see rollArr2D_to_Img for details, by default True.
    compress_colors : bool, optional
        used when returning images: see rollArr2D_to_Img for details, by default True.

    Returns
    -------
    'array_r3' mode:
        (stacked_roll_array,ins_codes) : (np.ndarray of shape=(notes,timesteps,instruments), tuple)
            tuple of: 3D piano roll made by padding & stacking 2D Instrument piano rolls from .mid file along timesteps axis and corresponding instrument code strings.
    'array_r2' mode:
        one_ins_roll: np.ndarray, shape=(notes,timesteps)
            2D piano roll made by using pretty_midi.get_piano_roll on the .mid file.
    'sep_roll'/'one_roll' mode:
        num_rolls : int
            Number of piano rolls created and saved to disk.
    """      
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
    # get the instrument code and the piano roll for all instruments in the midi file
    for ins in midi_container.instruments:
        code = instrument_to_insCode(ins)
        roll = []
        if ins.is_drum:
            # if instrument is a drum, use custom function to get the piano roll
            roll = drum_ins_to_roll(ins,fs=fs)
        else:
            # otherwise use the pretty_midi function on the instrument
            roll = ins.get_piano_roll(fs=fs)
        ins_codes.append(code)
        ins_rolls.append(roll)
    # get a separate roll using pretty_midi function on the entire midi file; flattening across instruments while ignoring drums
    one_ins_roll = midi_container.get_piano_roll(fs=fs)

    if out_type == 'array_r3':
        max_len = max([r.shape[1] for r in ins_rolls])
        padded_rolls = []
        for i in range(len(ins_rolls)):
            r = ins_rolls[i]
            padded_r = np.pad(r,[(0,0),(0,max_len-r.shape[1])]) # pad all instrument rolls to the same shape (timesteps)
            padded_rolls.append(padded_r)
        stacked_roll_array = np.stack(padded_rolls,axis=-1) # stack padded rolls into a single 3D piano roll
        return (stacked_roll_array,ins_codes)
    elif out_type == 'sep_roll':
        dest_splits = path_splitter(dest_path)
        for i in range(len(ins_rolls)):
            roll_array,roll_code = ins_rolls[i],ins_codes[i]            
            save_path = os.path.join(dest_splits['directory'],dest_splits['name']+"-ins_"+roll_code+".png")
            roll_img = rollArr2D_to_Img(roll_array,brighten=brighten,compress_colors=compress_colors) # convert each roll to a PIL Image..
            roll_img.save(save_path) # and save it to a separate file (name suffixed with the instrument code)
        return len(ins_rolls)
    else:
        # if not creating separate rolls for instruments, create a single roll for entire midi track with instrument prog=0 (Piano)
        if out_type == 'one_roll':
            roll_img = rollArr2D_to_Img(one_ins_roll,brighten=brighten,compress_colors=compress_colors)
            roll_img.save(dest_path)
            return 1
        elif out_type == 'array_r2':
            return one_ins_roll


def rollPic_slicer(source_path,dest_folder,fs=25,compress_colors=True,slice_dur=5,slice_suffix="-sp{:03d}"):
    """Cuts up a piano roll image into slices (images) [vertical cuts] that represent a fixed duration of .mid file audio.

    Parameters
    ----------
    source_path : str/os.path
        path to piano roll image.
    dest_folder : str/os.path
        path to directory where roll image slices are to be stored.
    fs : int, optional
        Sampling frequency used to generate the original roll image, by default 25.
    compress_colors : bool, optional
        whether compression across color channels was performed to generate the original roll image, by default True.
    slice_dur : int, optional
        maximum duration (in seconds) of each slice, by default 5.
    slice_suffix : str, optional
        formatting string used to name the slice image files, by default "-sp{:03d}".

    Returns
    -------
    im_counter : int
        Number of image slices created and saved to disk.
    """    
    src_splits = path_splitter(source_path)

    roll_img = Image.open(source_path)
    roll_img.load() # load the image from source_path as a PIL Image
    W,H = roll_img.size
    sl = slice_dur*fs # number of pixels that represent `slice_dur` seconds of audio
    if compress_colors:
        sl //= 3 # if color compression was used, 1/3-rd of the pixels represent `slice_dur` seconds of audio
    x = 0
    im_counter = 0
    while x < W:
        if x+sl > W:
            break
        dest_path = os.path.join(dest_folder,src_splits['name']+slice_suffix.format(im_counter)+".png") 
        crop_area = (x,0,x+sl,H) # tuple of coordinates for crop area
        chunk = roll_img.crop(crop_area) # crop the image
        chunk.load() 
        chunk.save(dest_path) # save the cropped image to disk
        im_counter += 1
        x += sl
    return im_counter


def roll_to_midi(source_path,dest_path,input_array=None,ins_codes=('0n',),fs=25,compress_colors=True): 
    """Converts a piano roll (image/raw array) into a .mid file. Internally transforms 2D np.ndarrays & image files to 3D np.ndarrays and converts them using rollArr3D_to_PrettyMIDI function.

    Parameters
    ----------
    source_path : str/os.path
        used for conversion from image: path to piano roll image.
    dest_path : str/os.path
        path to .mid file or to directory (if directory: use name of source .png file appended with '-resynth' for output .mid file).
    input_array : np.ndarray of shape=(notes,timesteps) or (notes,timesteps,instruments) or None, optional
        if not None, will use as roll array instead of reading it from image at source_path, by default None.
    ins_codes : tuple/list, optional
        String codes for the instruments used to generate the .mid file (see instrument_to_insCode for details), by default ('0n',) (setting for 2D np.ndarray & image rolls).
    fs : int, optional
        Sampling frequency: number of columns in input roll image/array per second of audio (each column is separated by 1/fs seconds), by default 25.
    compress_colors : bool, optional
        used for conversion from image: whether input roll image should be treated as if compression across color channels was performed to generate it, by default True.
    """    
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
        roll_img = Image.open(source_path)
        roll_img.load()
        roll_raw = np.asarray(roll_img,dtype='float32')
    # piano roll image is for only one instrument so it is first converted to 2D array of shape=(notes,timesteps)
    if compress_colors is True:
        roll_to_conv = roll_raw.reshape((roll_raw.shape[0],int(roll_raw.shape[1]*3))) # unpack the color channel values into the 2nd dimension if image treated as compressed
    elif compress_colors is False:
        roll_to_conv = np.mean(roll_raw,axis=2) # take the mean color across the 3 color channels if image treated as un-compressed
    elif compress_colors is None:
        roll_to_conv = roll_raw # use input_array as-is if it was input

    # for the conversion back to MIDI, roll_to_conv must be 3D array of shape=(notes,timesteps,instruments)
    if len(roll_to_conv.shape) < 3:
        roll_to_conv = roll_to_conv.reshape(roll_to_conv.shape+(1,)) # if originally 2D, number of instruments = 1

    def lim_val(val):
        val -= (val % 127) * 128
        return val
    
    roll_to_conv = np.vectorize(lim_val)(roll_to_conv) # values in the piano roll are limited to the range [0,127]

    roll_midi = rollArr3D_to_PrettyMIDI(roll_to_conv, ins_codes, fs=fs) # pass the 3D piano roll and the input ins_codes to function to get pretty_midi.PrettyMIDI object
    roll_midi.write(dest_path) # write the contents of the PrettyMIDI object to the .mid file specified in dest_path


def midi_to_rollTxt(source_path,dest_path,all_ins=True,fs=25,enc_fn=hex2):
    """Converts a .mid file to a .txt file containing an encoded form of its piano roll: notes+velocities to be played for each instrument per timestep. 
    Internally uses midi_to_roll function.
    If there are no notes to be played for an instrument for the timestep, a single null note is shown: 00-00.

    Parameters
    ----------
    source_path : str/os.path
        path to input .mid file.
    dest_path : str/os.path
        path to .txt file or to directory (if directory: use name of source .mid file for output .txt file).
    all_ins : bool, optional, by default True
        True- transcribe notes using their corresponding instruments specified in the .mid file:
            Template for each line of output .txt:
            <Instrument_0_code>: <Note>-<Velo> <Note>-<Velo> ...<TAB><Instrument_1_code>: <Note>-<Velo> <Note>-<Velo> ...<TAB><NEWLINE>
        False: transcribe notes in the .mid file without instrument information, the output txt file can only be read back with 1 instrument, set by default to Acoustic Grand Piano - instrument program 0.
            Template for each line of output .txt:
            <Note>-<Velo> <Note>-<Velo> <Note>-<Velo> <Note>-<Velo> ...<NEWLINE>
    fs : int, optional
        Sampling frequency: Number of lines of encoded text to generate per second of audio (each line is separated by 1/fs seconds), by default 25.
    enc_fn : function, optional
        specifies how notes, velocities, and instrument codes are to be encoded [see definition of hex2 function for an example], by default hex2.

    Returns
    -------
    num_timesteps : int
        Number of timesteps (seconds*fs) written to the .txt file.
    """    
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
    num_timesteps, num_instruments = 0, 1
    try:   
        if not all_ins:
            ## single instrument mode - roll to be transcribed omitting instrument information
            roll_array = midi_to_roll(source_path,"",'array_r2',fs=fs)
            roll_array = np.reshape(roll_array,roll_array.shape+(1,))
        else:
            ## all instruments mode
            roll_array,ins_codes = midi_to_roll(source_path,"",'array_r3',fs=fs)
    except Exception:
        print("failed to get roll_array")
        return 0

    num_timesteps = roll_array.shape[1] # get number of timesteps/columns from 2nd axis of roll_array    
    num_instruments = roll_array.shape[2] # get number of instruments from 3rd axis of roll_array 

    if all_ins:
        # ins_code is of the form: <number from 0-127><n/d based on is_drum>
        # encode the number part of the ins_codes using the encoding function enc_fn
        encoded_ins_codes = [enc_fn(int(c[:-1]),"forward")+c[-1] for c in ins_codes]

    for t in range(num_timesteps):
        timestep = roll_array[:,t,:]
        timestep_str = ""
        for i in range(num_instruments): # will loop only once if all_ins is False
            instrument = timestep[:,i]
            non_zero_notes = list(np.nonzero(instrument)[0]) #indices of notes with non-zero velocity values
            ins_str = encoded_ins_codes[i]+": " if all_ins else "" # include instrument code only if all_ins is True

            if len(non_zero_notes) == 0:
                # if no notes for instrument, add a null note: 00-00
                ins_str += enc_fn(int(0),'forward')+'-'+enc_fn(int(0),'forward')+" "
            else:
                # otherwise, add note-velo strings separated by spaces
                for n in non_zero_notes:
                    ins_str += enc_fn(int(n),'forward')+'-'+enc_fn(int(instrument[n]),'forward')+" "
            
            ins_str += '\t' if all_ins else "" # tabs used to separate instruments only if all_ins is True
            timestep_str += ins_str
        timestep_str += '\n' # add newline after processing the timestep (column of piano roll)
        txt_out.write(timestep_str)

    # after writing all events to the file, write a newline and close the text object.
    txt_out.write("\n")
    txt_out.close()
    return num_timesteps


def rollTxt_to_midi(source_path,dest_path,fs=25,dec_fn=hex2,logging=True):
    """Converts a .txt file containing an encoded form of a MIDI piano roll (notes+velocities, optionally instrument codes) to a .mid file.
    Internally uses roll_to_midi function. Expects formatting as described in the function midi_to_rollTxt.

    Parameters
    ----------
    source_path : str/os.path
        path to input .txt file.
    dest_path : str/os.path
        path to output .mid file or to directory (if directory: use name of source .txt file for output .mid file).
    fs : int, optional
        Sampling frequency: number of lines of encoded text to convert for producing 1 second of audio (each line is separated by 1/fs seconds), by default 25.
    dec_fn : function, optional, by default hex2
        function that specifies how the notes, velocities and instrument codes are to be decoded [see definition of hex2 for an example].
    logging : bool, optional
        whether or not to output statistics on the conversion process through print and to a log file, by default True.
    """    
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".mid")
        dest_splits = path_splitter(dest_path)
    
    # try to read the text file
    try:
        txt_in = open(source_path,'r')
    except IOError:
        print("File at {} does not exist... Exiting program".format(source_path))
        return None
    
    # create regexes which will be used later to find instrument codes, notes & velocities from the lines of the .txt file
    inst_regex, note_velo_regex = "", ""
    if dec_fn == hex2:
        # if decoding with hex2: notes, velocities, instrument codes (values in the range [0,127]) will be exactly 2 characters long
        inst_regex = r".{2,2}[dn]:"
        note_velo_regex = r".{2,2}-.{2,2}"
    elif dec_fn == None:
        # if no decoding required, the above values can be 1-3 characters long
        inst_regex = r".{1,3}[dn]:"
        note_velo_regex = r".{1,3}-.{1,3}"
    inst_regex, note_velo_regex = re.compile(inst_regex), re.compile(note_velo_regex)

    raw_lines = txt_in.readlines() # read the .txt file
    instrument_rolls = {} # to store the piano rolls of the instruments identified in the .txt file

    # variables used to log details of the conversion process
    instrument_event_counts = {}
    events_processed = 0

    for t in range(len(raw_lines)):
        line = raw_lines[t]
        inst_splits = [s for s in line.split('\t') if s != '\n' and s != ''] # expects events for each instrument to be separated by tabs
        for inst_line in inst_splits:
            inst_str = inst_regex.findall(inst_line)
            note_velo_strs = note_velo_regex.findall(inst_line)
            if not note_velo_strs: # if no valid notes are found, skip this line.
                continue
            if inst_str: # if there are matches to the regex, take the first match as the inst_code (encoded) after excluding the ":" character
                inst_code = inst_str[0][:-1]
            else: # else, use a default instrument -> this will occur for text files generated with midi_to_rollTxt with the setting all_ins=False
                inst_code = "0n"
            
            # then decode the program number from the inst_code as needed
            if dec_fn:
                inst_code = str(dec_fn(inst_code[:-1],"reverse"))+inst_code[-1]
            # if inst_code is invalid, use the default instrument
            if not check_insCode(inst_code):
                inst_code = "0n"
            
            # if the inst_code hasn't been seen before, create a 2D piano roll for it
            if inst_code not in instrument_rolls.keys():
                instrument_rolls[inst_code] = np.zeros((128,len(raw_lines)))
                instrument_event_counts[inst_code] = 0
            
            for n_v in note_velo_strs: # strs matching the regex are of the form: <Note>-<Velocity>
                n,v = n_v.split('-') # first separate note and velocity strings
                # then decode the strings back to integers
                if dec_fn: 
                    # if decoding function is given, decode using it.
                    n,v = dec_fn(n,"reverse"), dec_fn(v,"reverse")
                else:
                    # otherwise, attempt to directly convert the strings to base-10 integers
                    n,v = int(n), int(v)
                
                instrument_rolls[inst_code][n,t] = v # update the 2D piano roll for the instrument
                # increment the logging variables
                instrument_event_counts[inst_code] += 1
                events_processed += 1
    
    # collate instrument codes and their corresponding piano rolls into separate lists
    instrument_codes = []
    list_of_rolls = []
    for code, roll in instrument_rolls.items():
        instrument_codes.append(code)
        list_of_rolls.append(roll)
    
    # stack the 2D piano rolls of the instruments into a 3D piano roll
    stacked_roll_array = np.stack(list_of_rolls,axis=-1)
    # convert the 3D piano roll to a midi file through roll_to_midi
    roll_to_midi(source_path,dest_path,input_array=stacked_roll_array,ins_codes=instrument_codes,fs=fs)
    
    if logging: # Print logs and write them to a log file if necessary
        note_count_logs = ["Instrument {}: {} events".format(code, event_count) for code,event_count in instrument_event_counts.items()]
        log_strings = [
            '-------------------------------------------------',
            "Converted rollTxt {} to midi!".format(source_path),
            '*************************************************',
            "Transcribed {} events for {} instruments".format(events_processed,len(instrument_codes)),
            '-------------------------------------------------',
        ]
        log_strings += note_count_logs
        log_path = os.path.join(dest_splits['directory'],src_splits['name']+"-rollTxt_conv_log.txt")
        log_file = open(log_path,'a+')
        for line in log_strings:
            print(line)
        log_file.writelines(line+"\n" for line in log_strings) # write to log file in the same directory as the output midi file
        log_file.close()


def midi_to_csv(source_path,dest_path,ret_csv_strings=False):
    """Converts a .mid file to a .csv file using py-midicsv - https://pypi.org/project/py-midicsv/

    Parameters
    ----------
    source_path : str/os.path
        path to input .mid file.
    dest_path : str/os.path
        path to .csv file or to directory (if directory: use name of source .mid file for output .csv file).
    ret_csv_strings : bool, optional
        True: return list of csv formatted strings to calling function, False: write the list to a .csv file specified by dest_path, by default False.

    Returns
    -------
    csv_str_list : list
        list of csv formatted strings containing the instructions of the MIDI file; only returned if input ret_csv_strings is True.
    """    
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


def csv_to_midi(source_path,dest_path):
    """Converts a .csv file to a .mid file using py-midicsv - https://pypi.org/project/py-midicsv/

    Parameters
    ----------
    source_path : str/os.path
        path to input .csv file.
    dest_path : str/os.path
        path to .mid file or to directory (if directory: use name of source .csv file for output .mid file).
    """    
    src_splits = path_splitter(source_path)
    dest_splits = path_splitter(dest_path)

    if dest_splits['extension'] == '':
        # if dest_path points to a directory and not a .mid file
        # default to using source file name for output file
        dest_path = os.path.join(dest_path,src_splits['name']+".mid")
    midi_obj = py_midicsv.csv_to_midi(source_path)
    
    with open(dest_path,"wb") as out_midi:
        midi_writer = py_midicsv.FileWriter(out_midi)
        midi_writer.write(midi_obj)
    

spec_analysis_options = '--quiet --analysis -min 27.5 -max 19912.127 --bpo 12 --pps 25 --brightness 1'
wav_sine_synth_options = '--quiet --sine -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'
wav_noise_synth_options = '--quiet --noise -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'

def wav_to_spectro(source_path,dest_path,options=spec_analysis_options,encode=True):
    """Converts a .wav file to a spectrogram (.png file) using ARSS - http://arss.sourceforge.net

    Parameters
    ----------
    source_path : str/os.path
        path to input .wav file.
    dest_path : str/os.path
        path to output spectrogram (.png file) or to directory (if directory: use name of source .wav file for output .png file).
    options : str, optional
        space-separated string of command line options passed to ARSS: selects analysis mode, frequency range, beats per octave, pixels/sec, etc., 
        by default spec_analysis_options = '--quiet --analysis -min 27.5 -max 19912.127 --bpo 12 --pps 25 --brightness 1'
    encode : bool, optional
        whether or not to re-encode the generated .png spectrogram as rgba - appends '-enc' to output file name, by default True
    """    
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
    """Converts a spectrogram/image (.png file) to a .wav file using ARSS - http://arss.sourceforge.net

    Parameters
    ----------
    source_path : str/os.path
        path to input spectrogram/image (.png file)
    dest_path : str/os.path
        path to output .wav file or to directory (if directory: use name of source .png file for output .wav file)
    options : str, optional
        space-separated string of command line options passed to ARSS: selects synthesis mode (noise/sine), frequency range, beats per octave, pixels/sec, etc., 
        by default wav_noise_synth_options = '--quiet --noise -min 27.5 -max 19912.127 --pps 25 -r 44100 -f 16'
    """    
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