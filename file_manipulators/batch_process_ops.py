'''
Module with functions made to process large batches of files: send through series of conversions/operations
'''
# imports of built-in packages
import os
import glob
import shutil
import argparse # handles execution of batch operations from command line
import datetime

# imports from package modules
from .common_file_ops import path_splitter, ensure_dir, file_segmenter, metadata_remover
from .audio_conv_ops import *
# ------------------------------------------------------------------------------------------------- #
## Functions to process large batches of files 

def midi_to_arss_batch(midi_folder,arss_folder,split_size=5,delete_wav=True):
    """Takes a directory of .mid files and converts them to .wav files. 
    Then converts the .wav files into several ARSS spectrograms representing audio clips of `split_size` seconds.

    Parameters
    ----------
    midi_folder : str/os.path
        path to root directory containing .mid files (will recursively search in directory for .mid files).
    arss_folder : str/os.path
        path to directory where ARSS spectrogram slices will be stored.
    split_size : int, optional
        duration of ARSS spectrogram slices in seconds, by default 5.
    delete_wav : bool, optional
        whether or not to delete the intermediate .wav files, by default True.
    """    
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
    """Takes a directory of .mid files and converts them to piano roll image slices (.png images of rolls flattened across original instruments).

    Parameters
    ----------
    midi_folder : str/os.path
        path to root directory containing .mid files (will recursively search in directory for .mid files).
    rolls_folder : str/os.path
        path to directory where piano roll image slices will be stored.
    split_size : int, optional
        duration of piano roll image slices in seconds, by default 5.
    conv_resolution : int, optional
        Sampling frequency for piano roll columns (each column is separated by 1/conv_resolution seconds), by default 25.
    to_brighten : bool, optional
        determine pixel brightness range of output piano roll image slices: True=(0,255), False=(0,127), by default True.
    compress_colors : bool, optional
        whether or not piano roll columns are compressed using the 3 color channels of the image, by default True.
    """    
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # create suffixes for subfolder names and create the subfolders
    brightness_lvl = "(0-127)"
    if to_brighten:
        brightness_lvl = "(0-255)"
    f_prefix_full = "full, "
    f_prefix_seg = "t="+str(split_size)+", "
    f_suffix = "fs="+str(conv_resolution)+", b="+brightness_lvl
    if compress_colors:
        f_suffix +=" -cc"

    full_track_folder = os.path.join(rolls_folder,f_prefix_full+f_suffix)
    segment_folder = os.path.join(rolls_folder,f_prefix_seg+f_suffix)

    for p in [full_track_folder,segment_folder]:
        ensure_dir(p)
    
    # Starting from MIDI Files
    # 1. Convert all MIDI files in source path to midi rolls
    print("Creating piano roll images for full tracks:")
    midi_counter = 0
    failed_files,error_counter = [], 0
    midi_glob = glob.glob(midi_folder+"/**/*.mid", recursive=True)
    for filepath in midi_glob:
        conv = midi_to_roll(filepath,full_track_folder,'one_roll',fs=conv_resolution,brighten=to_brighten,compress_colors=compress_colors)
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
        chunk_counter += rollPic_slicer(filepath,segment_folder,fs=conv_resolution,compress_colors=compress_colors,slice_dur=split_size)
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

def midi_to_rollTxt_batch(midi_folder,txt_path,all_inst=False,conv_resolution=25,enc_fn=hex2):
    """Takes a directory of .mid files, converts them to text which encodes piano roll notes & velocities.
    Appends the text of all .mid files together in one large .txt file.
    
    Parameters
    ----------
    midi_folder : str/os.path
        path to root directory containing .mid files (will recursively search in directory for .mid files).
    txt_path : str/os.path
        path to .txt file which will store the text encodings of the .mid files.
    all_inst : bool, optional
        True: encode original instrument info, False: use default instrument - Acoustic Grand Piano for all notes, by default False.
    conv_resolution : int, optional
        number of text lines per second of audio (resolution), by default 25.
    enc_fn : function, optional
        function used to encode piano roll note & velocity values and instrument codes as strings, by default hex2 (from audio_conv_ops).
    """    
    ## bulk conversion of midi files to one large text file as encoded notes
    midi_counter = 0
    line_count = 0
    midi_glob = glob.glob(midi_folder+"/**/*.mid", recursive=True)
    for filepath in midi_glob:
        t = midi_to_rollTxt(filepath,txt_path,all_ins=all_inst,fs=conv_resolution,enc_fn=enc_fn)
        if t != 0:
            line_count += t
            midi_counter += 1
    print("Total midis: "+str(midi_counter))
    print("Total timesteps: "+str(line_count))
# ----------------------------------------------------------------------------------------- #
# Command line functionality for bulk conversion operations #
def batchOps_from_command_line():
    '''
    Function to provide command line functionality for bulk conversion operations.
    '''
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


    parser.add_argument('batch_conv_operation',type=str,choices=list(bco_choices.keys()),help=bco_help_msg)
    parser.add_argument('--src_dir','-i',type=str,help='Directory containing files to convert')
    parser.add_argument('--dest_dir','-o',type=str,help='Directory to save converted files')
    parser.add_argument('--chunk_size','-t',type=int,default=5,help= 'Duration for chunks (in secs) (default: %(default)s)')
    parser.add_argument('--conv_resolution','-r',type=int,default=25,help='<For piano rolls> Sampling frequency:\n time separation b/w columns = 1/resolution (default: %(default)s)')
    parser.add_argument('--use_all_instr','-ins',action='store_true',help='<For piano rollTxt> Use original instruments of midi file? (will use acoustic grand piano if option omitted)')
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
                midi_to_rollTxt_batch(src,dest,all_inst=args.use_all_instr,conv_resolution=args.conv_resolution)
            else:
                print("Invalid choice for batch_conv_operation...")
        
    except Exception as e:
        print("Exception occured, see traceback below:")
        print('-------------------------------------------------')
        print(e)
