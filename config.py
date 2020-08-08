'''
Module used to configure the paths to the executables & packages needed for the file_manipulators package
'''
import os
import json
import argparse

def write_config():
    '''
    Writes a new configuration file or overwrites an existing configuration file
    '''
    print("*************************************************************************************************")
    print("This package requires access to certain executables for some file conversion functions to work..")
    print("If an executable is not installed, functions using it will throw errors when run.")
    print("Enter the filepaths to the required executables as prompted")
    ffmpeg_path = input("FFmpeg: ffmpeg.exe\n")
    timidity_path = input("Timidity++: timidity.exe\n")
    waon_path = input("WaoN: waon.exe\n")
    arss_path = input("ARSS: arss.exe\n")
    print("-------------------------------------------------------------------------------------------------")
    print("This package also requires a clone of the pretty_midi/examples folder from:\n https://github.com/craffel/pretty-midi")
    print("After downloading the folder, enter the filepath to it below.")
    pretty_midi_examples_path = input("pretty_midi/examples folder path:\n")
    print("*************************************************************************************************")

    path_dict = {
        "FFMPEG_PATH": ffmpeg_path,
        "TIMIDITY_PATH": timidity_path,
        "WAON_PATH": waon_path,
        "ARSS_PATH": arss_path,
        "PRETTY_MIDI_EXAMPLES_PATH": pretty_midi_examples_path
    }

    config_path = os.path.join(os.path.dirname(__file__),"config.json")
    json_file = open(config_path,"w")
    json_file.write(json.dumps(path_dict,sort_keys=True,indent=3))
    print("Config file saved to "+config_path)

def read_config(print_config=False):
    '''
    Reads the configuration file saved in the package directory
    If it doesn't exist, sends an error message
    '''
    config_file = ""
    try:
        config_file = open(os.path.join(os.path.dirname(__file__), "config.json"), "r")
    except FileNotFoundError:
        print("Configuration file does not exist...")
        print("Please run on command line: python \"path/to/file_manipulators/config.py\" write")
        print("Alternatively, run the write_config() function of config.py")
    config = json.loads(config_file.read())
    if print_config:
        print("Saved configuration:")
        print("{")
        for k,v in config.items():
            print("{}: {}".format(repr(k),repr(v)))
        print("}")
    return config

# if called from command line
if __name__ == "__main__":
    cmdparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    cmdparser.add_argument('config_operation',type=str,choices=["write","read"],help="Write/Read configuration file")
    args = cmdparser.parse_args()
    if args.config_operation == 'write':
        print("Writing configuration file...")
        write_config()
    elif args.config_operation == 'read':
        read_config(print_config=True)   

