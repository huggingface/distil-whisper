import argparse
import os
import subprocess
import urllib.parse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

def install_flash_attention():
    try:
        try:
            print("trying pip command")
            os.system('pip install flash-attn --no-build-isolation')
        except:
            print("trying pip3 command")
            os.system('pip3 install flash-attn --no-build-isolation')
    except Exception as e:
        print(f" Encountered error while installing flash-attn: {e}")


def install_optimum():
    try:
        try:
            print("trying pip command")
            os.system('pip install --upgrade optimum')
        except:
            print("trying pip3 command")
            os.system('pip3 install --upgrade optimum')
    except Exception as e:
        print(f" Encountered error while installing optimum: {e}")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--audio', type=str, default=None)
    # parser.add_argument('--length', type=str, default='short')
    # parser.add_argument('--category', type=str, default=None)
    # parser.add_argument('--speed', type=str, default=None)
    parser.add_argument('--list', type=str, default=None)

    parser.add_argument('url', type=str)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--out', type=str, default='.')
    parser.add_argument('--flash', action='store_true')
    parser.add_argument('--bt', action='store_true')
    parser.add_argument('--spec', action='store_true')

    args = parser.parse_args()

    url = args.url

    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme in ['http', 'https']:
        # This is an internet URL, proceed with yt-dlp
        output_path = args.out if args.out else os.path.join(os.getcwd(), '%(title)s.%(ext)s')
        command = ['yt-dlp', '-x', '--audio-format', 'flac', '-o', output_path, url]
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

    result = pipe(sample)
    print(result["text"])

if __name__ == "__main__":
    main()