# Prints estimated pitch

import pyaudio
import numpy as np
import struct

FFT_SIZE = 2**12

# Audio parameters
WIDTH = 2 # 2 bytes per sample
CHANNELS = 1 # Mono
RATE = 44100 # Sampling frequency

# setup audio
p = pyaudio.PyAudio()
stream = p.open(format   = p.get_format_from_width(WIDTH),
				channels = CHANNELS,
				rate     = RATE,
				input    = True,
				output   = True,
				frames_per_buffer = 1024) # Should this be "hop" ?

# Returns numpy array from audio input. Blocks until input is valid.
def from_stream(num):
	data_in_bytes = stream.read(num, exception_on_overflow=True)
	return struct.unpack(str(num)+'h', data_in_bytes)

# Sends numpy array to audio output. Blocks until output is ready.
def to_stream(data, gain=1):
	data = data * gain
	data_bytes = struct.pack(str(len(data))+'h', *data.astype(int)) 
	stream.write(data_bytes, len(data), exception_on_underflow=True) 

try:
	while True:
		samples = np.array(from_stream(FFT_SIZE)) # Get input
		# Need to window?
		freqs = np.fft.fft(samples) # FFT
		mags = np.abs(freqs) # Get magnitudes
		maxFreq = np.where(mags==max(mags))[0][0] # Find max freq
		if maxFreq >= FFT_SIZE / 2: # Make negative if necessary
			maxFreq = maxFreq - FFT_SIZE
		maxFreq_hertz = maxFreq * RATE / FFT_SIZE # convert to Hertz
		print('bin:',maxFreq,'\t','Freq:',maxFreq_hertz,'Hz') # print output
		to_stream(samples) # Pass input to audio output
		
except KeyboardInterrupt:
	stream.stop_stream()
	stream.close()
	p.terminate()
	print("--- done -----------------------")
