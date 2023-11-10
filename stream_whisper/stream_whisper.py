import argparse
import os
import subprocess
import urllib.parse
import platform
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
# from datasets import load_dataset
import configparser
import time
import filetype
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style, init
import random
from importlib.metadata import distribution, PackageNotFoundError
import re


init(autoreset=True)

art = """       .-.                                            .-.    _                        
      .' `.                                           : :   :_;                       
 .--. `. .'.--.  .--.  .--.  ,-.,-.,-. _____ .-..-..-.: `-. .-. .--. .---.  .--. .--. 
`._-.' : : : ..'' '_.'' .; ; : ,. ,. ::_____:: `; `; :: .. :: :`._-.': .; `' '_.': ..'
`.__.' :_; :_;  `.__.'`.__,_;:_;:_;:_;       `.__.__.':_;:_;:_;`.__.': ._.'`.__.':_;  
                                                                     : :              
                                                                     :_;              """

colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

for char in art:   
    print(random.choice(colors) + char, end="")


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(f"New file {event.src_path} has been detected.")
        kind = filetype.guess(event.src_path)
        if kind is None:
            print('Cannot guess file type!')
            return
        if kind.extension in ['mp3', 'flac']:
            global new_file_path
            new_file_path = event.src_path
            global observer
            observer.stop()

""" def print_custom_help():
    print(art) """

def check_ffmpeg():
    try:
        subprocess.check_output(['ffmpeg', '-version'])
        print("ffmpeg is installed.")
    except Exception as e:
        print(f"An error occurred while checking ffmpeg: {str(e)}")
        print("ffmpeg is not installed.")
        install_ffmpeg()

""" def install_ffmpeg():
    try:
        if platform.system() == 'Windows':
            print("Please install ffmpeg manually on Windows.")
        elif platform.system() == 'Linux':
            subprocess.check_call(['sudo', 'apt-get', 'install', 'ffmpeg'])
        elif platform.system() == 'Darwin':
            subprocess.check_call(['brew', 'install', 'ffmpeg'])
    except Exception as e:
        print(f"An error occurred while installing ffmpeg: {str(e)}")
 """

def install_ffmpeg():
    try:
        subprocess.check_output(['ffmpeg', '-version'])
        print("ffmpeg is installed.")
    except Exception as e:
        print("ffmpeg is not installed.")
        if platform.system() == 'Windows':
            print("Please install ffmpeg manually on Windows. You can download it from https://ffmpeg.org/download.html")
        elif platform.system() == 'Linux':
            print("Please install ffmpeg manually on Linux. On Debian/Ubuntu, you can use 'sudo apt install ffmpeg'. On Fedora, you can use 'sudo dnf install ffmpeg'. On Arch Linux, you can use 'sudo pacman -S ffmpeg'.")
        elif platform.system() == 'Darwin':
            print("Please install ffmpeg manually on macOS. If you have Homebrew installed, you can use 'brew install ffmpeg'.")

""" 
def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False """
    
def is_installed(package_name):
    try:
        distribution(package_name)
        return True
    except PackageNotFoundError:
        return False
    

def install_flash_attention():
    if not is_installed('flash-attn'):
        try:
            print("trying pip command")
            os.system('pip install flash-attn --no-build-isolation')
        except:
            print("trying pip3 command")
            os.system('pip3 install flash-attn --no-build-isolation')
    else:
        print("checking if flash-attn is installed...")
        print("flash-attn is already installed")

def install_optimum():
    if not is_installed('optimum'):
        try:
            print("trying pip command")
            os.system('pip install --upgrade optimum')
        except:
            print("trying pip3 command")
            os.system('pip3 install --upgrade optimum')
    else:
        print("checking if  optimum is installed...")
        print("optimum is already installed")

def determine_audio_length(audio_path):
    # Check if ffmpeg is installed

    # Get file size in MB
    file_size = os.path.getsize(audio_path) / (1024 * 1024)

    # Load audio file and get its length in seconds
    audio = AudioSegment.from_file(audio_path)
    length_in_sec = len(audio) / 1000

    # Determine if the audio is short or long
    if file_size < 1 and length_in_sec < 30:
        return 'short'
    else:
        return 'long'


""" def download_audio(url, output_path):
    command = ['yt-dlp', '-x', '-f', 'bestaudio', '--audio-format', 'flac', '-o', output_path, url]
    subprocess.run(command, check=True) """

def download_audio(url, output_path):
    command = ['yt-dlp', '-x', '-f', 'bestaudio', '--audio-format', 'flac', '-o', output_path, url]
    subprocess.run(command, check=True)

def process_url(url, args):
    parsed_url = urllib.parse.urlparse(url)
    global new_file_path
    global sample

    new_file_path = None

    def sanitize_path(path):
        # Remove spaces and other non-alphanumeric characters
        return re.sub(r'\W+', '', path)


    if parsed_url.scheme in ['http', 'https']:
        if args.out:
            output_path = sanitize_path(args.out)
            # Check if an extension is provided
            if not output_path.lower().endswith(('.mp3', '.flac')):
                # Strip the extension and add .flac
                output_path = os.path.splitext(output_path)[0] + '.flac'
        else:
            # Generate a unique filename
            output_path = os.path.join(os.getcwd(), sanitize_path('%(title)s.%(ext)s'))
    
        # output_path = args.out if args.out else os.path.join(os.getcwd(), '%(title)s.%(ext)s')
        download_audio(url, output_path)
        kind = filetype.guess(output_path)
        if kind is None:
            print('Cannot guess file type!')
            return
        if kind.extension not in ['mp3', 'flac']:
            # Attempt to convert to .flac
            try:
                command = ['ffmpeg', '-i', output_path, output_path + '.flac']
                subprocess.run(command, check=True)
                output_path = output_path + '.flac'
            except subprocess.CalledProcessError:
                print('Error converting file to .flac')
                config = configparser.ConfigParser()
                if args.settings:
                    config.read(args.settings)
                else:
                    config.read(os.path.dirname(os.path.realpath(__file__)) + '/settings.ini')

                if config.getboolean('DEFAULT', 'interactive', fallback=False):
                    print(f"Downloaded audio at {output_path} is corrupted. Please check or manually convert it to .mp3 or .flac.")
                    if input("Should I wait for you to do that? (yes/no) ").lower() == 'yes':
                        global observer
                        event_handler = MyHandler()
                        observer = Observer()
                        observer.schedule(event_handler, path='.', recursive=True)
                        observer.start()
                        try:
                            while observer.is_alive():
                                observer.join(1)
                        except KeyboardInterrupt:
                            observer.stop()
                        observer.join()
                        if new_file_path is not None:
                            output_path = new_file_path
        stream_middle = output_path
    elif os.path.isfile(url):
        stream_middle = url
    else:
        raise ValueError(f"Invalid file or http link: {url}")


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v2"

    if args.flash:
        print("using flash...")
        install_flash_attention()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True
        )
    elif args.bt:
        print("using bettertransformers")
        install_optimum()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model = model.to_bettertransformer()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

    model.to(device)

    print("bain-one")

    processor = AutoProcessor.from_pretrained(model_id)

    if args.spec:
        print("using speculative decoding")
        model_id = "openai/whisper-large-v2"
        assistant_model_id = "distil-whisper/distil-large-v2"
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        
        assistant_model.to(device)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            generate_kwargs={"assistant_model": assistant_model},
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        print("normal decoding enabled")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
    """ if args.out is not None:
        # dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]
    else:
        sample = stream_middle """

    # global sample = stream_middle

    sample = stream_middle

    if args.long:
        print("long sample decoding")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )   
    elif args.short:
        print("short sample decoding")
        print("your audio must strictly be less than 30 seconds for this to give useful results")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        print("automatic audio length detection activated")
        
        size = determine_audio_length(sample)

        if size == 'short':
            print("I have detected short audio length")
            pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
        elif size == 'long':
            print("I have detected long audio length")
            pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )     
        else:
            print("Unknown error during audio processing.")
            print("Please check your audio file")
            print("Otherwise you can specify the --long argument to save me the stress of automatic audio length detection.")

        print("bain-two")
        # print(sample)


    result = pipe(sample)
    print(result)
    print("bain-three")

    """ if args.text == 'cli':
        print(result["text"])
    elif args.text == 'both':
        print(result["text"])
        with open('output.txt', 'w') as output:
            output.write(result["text"])
    else:
        try:
            os.makedirs(os.path.dirname(args.text), exist_ok=True)
            with open(args.text, 'w') as output:
                output.write(result["text"])
        except Exception as e:
            print(f"Failed to write to output file: {e}") """


    if args.text is None or args.text == 'both':
        print(result["text"])
        with open('output.txt', 'w') as output:
            output.write(result["text"])
    elif args.text == 'cli':
        print(result["text"])
    else:
        try:
            os.makedirs(os.path.dirname(args.text), exist_ok=True)
            with open(args.text, 'w') as output:
                output.write(result["text"])
        except Exception as e:
            print(f"Failed to write to output file: {e}")



def main():
    print("initializing stream-whisper...")
    check_ffmpeg()

    try:
        # parser.add_argument('--audio', type=str, default=None)
        # parser.add_argument('--length', type=str, default='short')
        # parser.add_argument('--category', type=str, default=None)
        # parser.add_argument('--speed', type=str, default=None)

        # parser.add_argument('url', type=str)
        # parser.add_argument('--short', action='store_true')
        # parser = argparse.ArgumentParser(add_help=False)
        parser = argparse.ArgumentParser(description='stream-whisper v1.0. Visit https://github.com/nathfavour/stream-whisper for source code')
        parser.add_argument('--long', action='store_true', help='use chunked processing for long audio')
        parser.add_argument('--short', action='store_true', help='used for audio less than 30 seconds.')
        parser.add_argument('--text', action='store_true', help='path to store transcribed audio')
        parser.add_argument('--flash', action='store_true', help='use flash for speed processing')
        parser.add_argument('--bt', action='store_true', help='use better transformers')
        parser.add_argument('--spec', action='store_true', help='use speculative decoding processing')
        # parser.add_argument('-h', action='store_true', help='Show this help message and exit')
        parser.add_argument('--settings', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings.ini'), help='Path to settings file')


        # parser.add_argument('--in', type=str, default=get_default_config().get('in', 'audio.mp3'), help='Input file')
        # parser.add_argument('--out', type=str, default=get_default_config().get('out', 'cli'), help='Output mode')
        # parser.add_argument('--size', type=str, default=get_default_config().get('size', 'short'), help='Size mode')
        # parser.add_argument('--distil', type=str, default=get_default_config().get('distil', 'auto'), help='Distil mode')


        parser.add_argument('--list', type=str, default=None, help='file containing list of urls or file paths of audio to be (downloaded before) transcribed')
        parser.add_argument('url', type=str, nargs='?', default=None, help='the link is the only required argument👍🏽')
        parser.add_argument('--out', type=str, default='both', help='Audio output path.')

        args = parser.parse_args()

        """ if args.help or not any(vars(args).values()):
            print_custom_help()
            parser.exit()
 """
        """ if not any(vars(args).values()):
            # print_custom_help()
            parser.print_help()
            parser.exit() """


        # url = args.url

        
        try:
            if args.list:
                with open(args.list, 'r') as f:
                    urls = f.read().splitlines()
                for url in urls:
                    process_url(url, args)
            elif args.url:
                process_url(args.url, args)
            else:
                print("Please provide a URL or a list file of URLs.")
        except Exception as e:
            print(e)

    except Exception as e:
        print(e)
        # print_custom_help()
        # parser.print_help()
        # parser.print_help()
        # parser.exit()


if __name__ == "__main__":
    main()





