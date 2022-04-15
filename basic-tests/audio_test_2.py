import pyaudio
import numpy as np
import struct

CHUNK = 1024
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
	channels=CHANNELS,
	rate=RATE,
	input=True,
	output=True,
	frames_per_buffer=CHUNK)

print("* recording...")
print('input  latency = ',stream.get_input_latency())
print('output latency = ',stream.get_output_latency())

try:
	while True:
		# Prepare data and fft
		data_in_bytes = stream.read(CHUNK)
		data_in_int = struct.unpack(str(CHUNK)+'h',data_in_bytes)
		freqs_in_rect = np.fft.fft(data_in_int)
		
		# Perform computation/edit ---------------
		freqs_out_rect = np.roll(freqs_in_rect,0)
		# ----------------------------------------
		
		# IFFT and package data for playback
		data_out_int = np.fft.ifft(freqs_out_rect).real.astype(int)
		data_out_bytes = b''
		for samp in data_out_int:
			data_out_bytes += struct.pack('h',samp)
		stream.write(data_out_bytes, CHUNK, exception_on_underflow=True)

except KeyboardInterrupt: # press ctl-C to exit.
	print("* done.")
	stream.stop_stream()
	stream.close()
	p.terminate()
