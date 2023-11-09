import argparse
import os
import subprocess
import urllib.parse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url', type=str)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--out', type=str, default='.')
    parser.add_argument('--flash', action='store_true')
    parser.add_argument('--bt', action='store_true')
    parser.add_argument('--spec', action='store_true')

    args = parser.parse_args()

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

    # Handle other processes here
    # ...

if __name__ == "__main__":
    main()