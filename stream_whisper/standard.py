import argparse
import os
import subprocess
import urllib.parse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from datasets import load_dataset
import platform
from pydub import AudioSegment
import pkg_resources

print("initializing stream-whisper...")

try:
    subprocess.check_output(['ffmpeg', '-version'])
    print("initialization complete")
except subprocess.CalledProcessError:
    # If ffmpeg is not installed, try to install it
    try:
        if platform.system() == 'Windows':
            print("Please install ffmpeg manually on Windows.")
        elif platform.system() == 'Linux':
            subprocess.check_call(['sudo', 'apt-get', 'install', 'ffmpeg'])
        elif platform.system() == 'Darwin':
            subprocess.check_call(['brew', 'install', 'ffmpeg'])
    except Exception as e:
        print("An error occurred while trying to install ffmpeg: ", e)
        print("Please install ffmpeg manually.")
        print("Otherwise you can specify the --long argument to save me the stress of automatic audio length detection.")

def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
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


""" def determine_audio_length(audio_path):
    # Implement your algorithm here
    # Return 'short' or 'long'
    pass
 """
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


def download_audio(url, output_path):
    command = ['yt-dlp', '-x', '-f', 'bestaudio', '--audio-format', 'flac', '-o', output_path, url]
    subprocess.run(command, check=True)


def process_url(url, args):
    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme in ['http', 'https']:
        output_path = args.out if args.out else os.path.join(os.getcwd(), '%(title)s.%(ext)s')
        download_audio(url, output_path)
        stream_middle = output_path
    elif os.path.isfile(url):
        stream_middle = url
    else:
        raise ValueError(f"Invalid URL: {url}")


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--audio', type=str, default=None)
    # parser.add_argument('--length', type=str, default='short')
    # parser.add_argument('--category', type=str, default=None)
    # parser.add_argument('--speed', type=str, default=None)

    # parser.add_argument('url', type=str)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--text', action='store_true')
    # parser.add_argument('--out', type=str, default='.')
    parser.add_argument('--flash', action='store_true')
    parser.add_argument('--bt', action='store_true')
    parser.add_argument('--spec', action='store_true')

    parser.add_argument('--list', type=str, default=None)
    parser.add_argument('url', type=str, nargs='?', default=None)
    parser.add_argument('--out', type=str, default='.')

    args = parser.parse_args()

    url = args.url

    parsed_url = urllib.parse.urlparse(url)

    if args.list:
        with open(args.list, 'r') as f:
            urls = f.read().splitlines()
        for url in urls:
            process_url(url, args)
    elif args.url:
        process_url(args.url, args)
    else:
        print("Please provide a URL or a list of URLs.")


    # yt-dlp -f bestaudio --extract-audio --audio-format flac <video_url>

    if parsed_url.scheme in ['http', 'https']:
        # This is an internet URL, proceed with yt-dlp
        output_path = args.out if args.out else os.path.join(os.getcwd(), '%(title)s.%(ext)s')
        command = ['yt-dlp', '-x', '-f', 'bestaudio', '--audio-format', 'flac', '-o', output_path, url]
        
        subprocess.run(command, check=True)
        stream_middle = output_path
    elif os.path.isfile(url):
        # This is a file URL, skip yt-dlp
        stream_middle = url
    else:
        raise ValueError(f"Invalid URL: {url}")


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v2"

    if args.speed == 'flash':
        install_flash_attention()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True
        )
    elif args.speed == 'bt':
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

    processor = AutoProcessor.from_pretrained(model_id)

    if args.category == 'spec':
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
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
    if args.audio is None:
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]
    else:
        sample = args.audio

    if args.length == 'long':
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
        size = determine_audio_length(in_file)

        if size == 'short':
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


    result = pipe(sample)
    print(result["text"])

if __name__ == "__main__":
    main()