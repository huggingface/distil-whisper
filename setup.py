from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='stream_whisper',
    version='0.1.0',
    url='https://github.com/nathfavour/stream-whisper',
    author='nathfavour',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=required,
    entry_points={
        'console_scripts': [
            'stream-whisper=stream_whisper.stream_whisper:main',
        ],
    },
)