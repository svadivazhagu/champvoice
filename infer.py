import sounddevice as sd
import numpy as np
import torch
import queue
import threading
from scipy.signal import resample
from model import Net
import json
import argparse

class AudioProcessor:
    def __init__(self, checkpoint_path, config_path, input_device=None, output_device=None, chunk_factor=1):
        # Load model and config
        with open(config_path) as f:
            config = json.load(f)
        self.model = Net(**config['model_params'])
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['model'])
        self.model.eval()

        self.sr = config['data']['sr']
        self.chunk_factor = chunk_factor
        self.L = self.model.L
        self.chunk_len = self.model.dec_chunk_size * self.L * chunk_factor

        # Initialize buffers
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, torch.device('cpu'))
        self.convnet_pre_ctx = (self.model.convnet_pre.init_ctx_buf(1, torch.device('cpu')) 
                               if hasattr(self.model, 'convnet_pre') else None)

        # Audio queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Audio device settings
        self.input_device = input_device
        self.output_device = output_device

    def process_audio(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(status)
        # Resample the input data to match the model's sample rate
        resampled_data = resample(indata[:, 0], int(len(indata) * self.sr / 48000))
        audio_chunk = torch.from_numpy(resampled_data).float()
        self.input_queue.put(audio_chunk)

    def audio_output_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        if status:
            print(status)
        try:
            audio_chunk = self.output_queue.get_nowait()
            # Resample the output data back to the output device's sample rate
            resampled_data = resample(audio_chunk, frames)
            outdata[:] = resampled_data.reshape(-1, 1)
        except queue.Empty:
            outdata.fill(0)

    def process_stream(self):
        """Process audio chunks from input queue"""
        while True:
            try:
                audio = self.input_queue.get()
                
                if len(audio) < self.chunk_len:
                    audio = torch.nn.functional.pad(audio, (0, self.chunk_len - len(audio)))
                
                with torch.inference_mode():
                    output, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = (
                        self.model(
                            audio.unsqueeze(0).unsqueeze(0),
                            self.enc_buf, self.dec_buf, self.out_buf,
                            self.convnet_pre_ctx,
                            pad=(not self.model.lookahead)
                        )
                    )
                
                self.output_queue.put(output.squeeze().numpy())
                
            except queue.Empty:
                continue

    def get_compatible_devices(self):
        """Get lists of compatible input and output devices"""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        # Find the WASAPI host API index
        wasapi_index = None
        for i, api in enumerate(hostapis):
            if api['name'] == 'Windows WASAPI':
                wasapi_index = i
                break

        wasapi_inputs = []
        wasapi_outputs = []

        print("\nAvailable WASAPI devices:")
        for i, device in enumerate(devices):
            if device['hostapi'] == wasapi_index:  # Check if device uses WASAPI
                if device['max_input_channels'] > 0:
                    print(f"Input #{i}: {device['name']}")
                    wasapi_inputs.append(i)
                if device['max_output_channels'] > 0:
                    print(f"Output #{i}: {device['name']}")
                    wasapi_outputs.append(i)

        if not wasapi_inputs or not wasapi_outputs:
            print("\nWARNING: No WASAPI devices found. Available Host APIs:")
            for i, api in enumerate(hostapis):
                print(f"API #{i}: {api['name']}")

        return wasapi_inputs, wasapi_outputs

    def print_available_apis(self):
        """Print all available audio APIs"""
        hostapis = sd.query_hostapis()
        print("\nAvailable Audio APIs:")
        for i, api in enumerate(hostapis):
            print(f"API #{i}: {api['name']}")
            # Print default devices for this API if they exist
            if api['default_input_device'] is not None:
                input_name = sd.query_devices(api['default_input_device'])['name']
                print(f"  Default Input: {input_name}")
            if api['default_output_device'] is not None:
                output_name = sd.query_devices(api['default_output_device'])['name']
                print(f"  Default Output: {output_name}")

    def start_streaming(self):
        """Start audio streaming"""
        # Print available APIs first
        self.print_available_apis()

        # Get compatible devices
        input_devices, output_devices = self.get_compatible_devices()

        if self.input_device is None or self.output_device is None:
            print("\nPlease select from the WASAPI devices listed above:")
            if self.input_device is None:
                self.input_device = int(input("Enter input device number: "))
            if self.output_device is None:
                self.output_device = int(input("Enter output device number (VB-Audio Virtual Cable): "))

        # Get device info
        input_device_info = sd.query_devices(self.input_device)
        output_device_info = sd.query_devices(self.output_device)

        print(f"\nSelected devices:")
        print(f"Input: {input_device_info['name']} using Windows WASAPI")
        print(f"Output: {output_device_info['name']} using Windows WASAPI")

        # Start processing thread
        processing_thread = threading.Thread(target=self.process_stream)
        processing_thread.daemon = True
        processing_thread.start()

        # Start audio streams
        input_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            callback=self.process_audio,
            samplerate=48000  # Use device's native sample rate
        )

        output_stream = sd.OutputStream(
            device=self.output_device,
            channels=1,
            callback=self.audio_output_callback,
            samplerate=48000  # Use device's native sample rate
        )

        with input_stream, output_stream:
            print("\nStreaming started. Press Ctrl+C to stop.")
            input_stream.start()
            output_stream.start()
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\nStreaming stopped.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '-p', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', '-c', type=str, required=True,
                       help='Path to model config')
    parser.add_argument('--input_device', '-i', type=int,
                       help='Input device number')
    parser.add_argument('--output_device', '-o', type=int,
                       help='Output device number (VB-Audio Virtual Cable)')
    parser.add_argument('--chunk_factor', '-n', type=int, default=1,
                       help='Chunk factor for streaming')

    args = parser.parse_args()

    processor = AudioProcessor(
        args.checkpoint_path,
        args.config_path,
        args.input_device,
        args.output_device,
        args.chunk_factor
    )
    processor.start_streaming()

if __name__ == '__main__':
    main()
