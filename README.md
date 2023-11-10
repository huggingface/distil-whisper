
## STREAM WHISPER

Well, I figured out distil-whisper would work better with stream capabilities, i.e ability to transcribe videos/audios directly from the internet, not just file paths. So, stream-whisper was born.

In case you still haven't figured, stream whisper is a fork of huggingface's distil whisper which is fundamentally a fast speech recognition and transcribing engine...(the devil is in the details, lol)

## INSTALLATION

1. Clone or download this repository
2. install the requirements with pip:

   
		pip install requirements.txt
 NB: If this fails, try `pip3 install requirements.txt` instead.
3. from the root directory, run:

        cd stream_whisper
   
## USAGE
the only mandatory argument is  <url> argument (file url or link url).

    python stream_whisper.py <url>
  
explicitly specify the output file:

		python run.py <url> --out '/output/path/of/audio'

specify the output the of the (transient) audio
N.B: This is only required for a streamed (downloaded) audio. if the audio is a local audio, the `out` argument is totally unnecessary and might lead to errors
  
		python run.py <url>  --out '/output/path/of/audio'

specify the output path of the transcription with --text. specifically, --text 'cli' and not specifying --text at all, or --text 'both' will output to both cli and a default file 'output.txt' in working directory.
meanwhile, any other value, e.g --text 'texter' or --text '/path/to/output/file' will output to the file path argument.

		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output'

specify --long argument to specify audio length, for chunking while processing. otherwise, automatic length detection is used.

		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --long

specify --short argument for audio lesser than 30 seconds length. otherwise, automatic length detection is used.
  
		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --short

# Additional Speed & Memory Improvements

Specify --spec to use Speculative Decoding
  
		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --spec

specify --flash to use Flash Attention

		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --flash

specify --bt to use better transformer

		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --bt

specify --settings to use a custom settings file.

		python run.py <url> --out '/output/path/of/audio' --text '/transcription/output' --settings '/path/to/settings/file'

N.B: The original README.md of distil whisper is [here](Original-README.md).


## TO-DO
- [ ] add to pip
- [ ] further tests
- [ ] further features



## Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v2) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program for Cloud TPU v4s

## Citation

If you use this model, please consider citing the Distil-Whisper paper:
```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

And also the Whisper paper:
```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```


