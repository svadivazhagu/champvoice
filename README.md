# Champion Voice
## *Still in development*
Champion Voice is a real-time voice conversion application that allows users to transform their voice during live conversations in applications like Discord. Built on the [LLVC (Low-Latency Voice Conversion)](https://arxiv.org/abs/2311.00873) architecture, it processes audio input through a virtual audio driver to provide seamless voice transformation with minimal latency.

## Features

- Real-time voice conversion using LLVC model architecture
- Low latency processing on CPU or GPU
- Integration with VBCable virtual audio driver
- Compatible with popular voice chat applications (Discord, Skype, etc.)
- Easy-to-use interface for voice transformation settings
- Multiple voice presets available

## Prerequisites

- Windows 10/11 64-bit
- VBCable Virtual Audio Driver ([Download Here](https://vb-audio.com/Cable/))
- Python 3.8 or higher
- CUDA Toolkit 12.4 (optional, for GPU support)

## Installation

1. Install VBCable Virtual Audio Driver
```bash
# Follow the installation instructions from VB-Audio website

git clone https://github.com/yourusername/champion-voice.git
cd champion-voice

curl -LsSf https://astral.sh/uv/install.sh | sh

# Cuda 12.4 support
uv sync --extra cu124
# Using CPU - not recommended
uv sync --extra cpu

## Inference
`python infer.py -p my_checkpoint.pth -c my_config.pth -f input_file -o my_out_dir` will convert a single audio file or folder of audio files using the given LLVC checkpoint and save the output to the folder `my_out_dir`. The `-s` argument simulate a streaming environment for conversion. The `-n` argument allows the user to specify the size of input audio chunks in streaming mode, trading increased latency for better RTF.

`compare_infer.py` allows you to reproduce our streaming no-f0 RVC and QuickVC conversions on input audio of your choice. By default, `window_ms` and `extra_convert_size` are set to the values used for no-f0 RVC conversion. See the linked paper for the QuickVC conversion parameters.

## Training
1. Create a folder `experiments/my_run` containing a `config.json` (see `experiments/llvc/config.json` for an example)
2. Edit the `config.json` to reflect the location of your dataset and desired architectural modifications
3. `python train.py -d experiments/my_run`
4. The run will be logged to Tensorboard in the directory `experiments/my_run/logs`

## Dataset
Datasets are comprised of a folder containing three subfolders: `dev`, `train` and `val`. Each of these folders contains audio files of the form `PREFIX_original.wav`, which are audio clips recorded by a variety of input speakers, and `PREFIX_converted.wav`, which are the original audio clips converted to a single target speaker. `val` contains clips from the same speakers as `test`. `dev` contains clips from different speakers than `test`. 